package pdss.engine

import org.apache.spark.HashPartitioner
import org.apache.spark.rdd.RDD
import pdss.core._   // SparseMatrix, DistVector, CSRMatrix, CSRRow, CSCMatrix, CSCCol
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast

import scala.reflect.ClassTag

/**
 * Baseline + optimized execution engine built only on RDDs.
 * - SpMV (COO × vector)
 * - SpMV (CSR × vector)
 * - SpMV (COO x Broadcast Dense Vector)
 * - SpMV (CSR x Broadcast Dense Vector)
 * - SpMM (COO × COO)
 * - SpMM (CSR × COO)
 * - SpMM (CSR × CSC)
 *
 * No DataFrames / Datasets.
 */
object ExecutionEngine {

  // ---------------------------------------------------------------------------
  // helper: co-partition two RDDs on Int key
  // ---------------------------------------------------------------------------
  private def coPartition[V: ClassTag, W: ClassTag](
      left: RDD[(Int, V)],
      right: RDD[(Int, W)]
  ): (RDD[(Int, V)], RDD[(Int, W)]) = {
    val p = new HashPartitioner(left.sparkContext.defaultParallelism * 2)
    val L = left.partitionBy(p).persist()
    val R = right.partitionBy(p).persist()
    (L, R)
  }

  // ---------------------------------------------------------------------------
  // 1) SpMV: y = A * x  (COO × SparseVector)
  // A: SparseMatrix in COO → entries: RDD[(i, j, v)]
  // x: DistVector (Sparse) → values: RDD[(j, xj)]
  // result: RDD[(i, yi)]
  // ---------------------------------------------------------------------------
  def spmv(A: SparseMatrix, x: DistVector): RDD[(Int, Double)] = {
    // key A by column j to join with x
    val AkeyedByJ: RDD[(Int, (Int, Double))] =
      A.entries.map { case (i, j, v) => (j, (i, v)) }

    val xByJ: RDD[(Int, Double)] = x.values

    val joined = AkeyedByJ.join(xByJ) // (j, ((i,v), xj))

    joined
      .map { case (_, ((i, v), xj)) => (i, v * xj) }
      .reduceByKey(_ + _)
  }

  // ---------------------------------------------------------------------------
  // 2) SpMV advanced: y = A * x  (CSR × SparseVector)
  //
  // A: CSRMatrix → rows: RDD[CSRRow(row, colIdx[], values[])]
  // x: DistVector (Sparse) → values: RDD[(j, xj)]
  // result: RDD[(i, yi)]
  // ---------------------------------------------------------------------------
  def spmvCSR(A: CSRMatrix, x: DistVector): RDD[(Int, Double)] = {
    // Key A by column k (which is the index j of vector x)
    // Maps to: (k, (i, vA))
    val AbyK: RDD[(Int, (Int, Double))] = A.rows.flatMap { row =>
      val i = row.row
      val cols = row.colIdx
      val vals = row.values
      
      // Emit (j, (i, vA)) for each non-zero element A(i,j)
      val buf = new scala.collection.mutable.ArrayBuffer[(Int, (Int, Double))](cols.length)
      var t = 0
      while (t < cols.length) {
        buf += ((cols(t), (i, vals(t)))) // (k, (i, vA)) where k=j
        t += 1
      }
      buf
    }

    // Vector x: RDD[(j, xj)]
    val xByJ: RDD[(Int, Double)] = x.values

    // Co-partition and join on k=j
    val (acp, xcp) = coPartition(AbyK, xByJ)

    // Joined: (j, ((i, vA), xj))
    val joined = acp.join(xcp)

    // Map to (i, vA * xj)
    joined
      .map { case (_, ((i, vA), xj)) => (i, vA * xj) }
      // Reduce by row index i to get final result yi
      .reduceByKey(_ + _)
  }

  // ---------------------------------------------------------------------------
  // 3) SpMV: y = A * x  (COO × DenseVector)
  //
  // Assumes x is a local Array[Double] broadcasted to all nodes.
  //
  // A: SparseMatrix in COO → entries: RDD[(i, j, v)]
  // x: Broadcast[Array[Double]] (Dense)
  // result: RDD[(i, yi)]
  // ---------------------------------------------------------------------------
  def spmvCooWithDense(A: SparseMatrix, x_bcast: Broadcast[Array[Double]]): RDD[(Int, Double)] = {
    val x = x_bcast.value
    A.entries
      .map { case (i, j, v) => 
        val xj = if (j < x.length) x(j) else 0.0 // Bounds check
        (i, v * xj) 
      }
      .reduceByKey(_ + _)
  }

  // ---------------------------------------------------------------------------
  // 4) SpMV advanced: y = A * x  (CSR × DenseVector)
  //
  // Assumes x is a local Array[Double] broadcasted to all nodes.
  // This is a highly efficient, shuffle-free SpMV implementation.
  //
  // A: CSRMatrix → rows: RDD[CSRRow(row, colIdx[], values[])]
  // x: Broadcast[Array[Double]] (Dense)
  // result: RDD[(i, yi)]
  // ---------------------------------------------------------------------------
  def spmvCsrWithDense(A: CSRMatrix, x_bcast: Broadcast[Array[Double]]): RDD[(Int, Double)] = {
    val x = x_bcast.value
    A.rows.map { row =>
      val i = row.row
      val cols = row.colIdx
      val vals = row.values
      
      var yi = 0.0
      var t = 0
      while (t < cols.length) {
        val j = cols(t)
        val vA = vals(t)
        // Perform local dot product
        if (j < x.length) { // Safety bounds check
          yi += vA * x(j)
        }
        t += 1
      }
      (i, yi)
    }
  }


  // ---------------------------------------------------------------------------
  // 5) SpMM baseline: COO × COO
  //
  // A(i,k,vA)  and  B(k,j,vB)
  // join on k  →  ((i,j), vA*vB)  → reduceByKey
  // ---------------------------------------------------------------------------
  def spmm(A: SparseMatrix, B: SparseMatrix): RDD[((Int, Int), Double)] = {
    val AbyK: RDD[(Int, (Int, Double))] =
      A.entries.map { case (i, k, vA) => (k, (i, vA)) }

    val BbyK: RDD[(Int, (Int, Double))] =
      B.entries.map { case (k, j, vB) => (k, (j, vB)) }

    val (acp, bcp) = coPartition(AbyK, BbyK)

    val joined = acp.join(bcp) // (k, ((i,vA),(j,vB)))

    val products = joined.map {
      case (_, ((i, vA), (j, vB))) =>
        ((i, j), vA * vB)
    }

    products.reduceByKey(_ + _)
  }

  // ---------------------------------------------------------------------------
  // 6) SpMM optimized: CSR (left) × COO (right)
  //
  // A: CSRMatrix → rows: RDD[CSRRow(row, colIdx[], values[])]
  // B: SparseMatrix (COO) → entries: RDD[(k, j, vB)]
  //
  // Strategy:
  //  - expand CSR rows to (k, (i, vA))
  //  - key B by k
  //  - join on k
  //  - emit ((i,j), vA*vB) and reduce
  // ---------------------------------------------------------------------------
  def spmmCSRWithCOO(A: CSRMatrix, B: SparseMatrix): RDD[((Int, Int), Double)] = {

    val AbyK: RDD[(Int, (Int, Double))] = A.rows.flatMap { row =>
      val i = row.row
      val cols = row.colIdx
      val vals = row.values
      val buf = new scala.collection.mutable.ArrayBuffer[(Int, (Int, Double))](cols.length)
      var t = 0
      while (t < cols.length) {
        buf += ((cols(t), (i, vals(t))))
        t += 1
      }
      buf
    }

    val BbyK: RDD[(Int, (Int, Double))] =
      B.entries.map { case (k, j, vB) => (k, (j, vB)) }

    val (acp, bcp) = coPartition(AbyK, BbyK)

    val joined = acp.join(bcp) // (k, ((i,vA),(j,vB)))

    val products = joined.map {
      case (_, ((i, vA), (j, vB))) =>
        ((i, j), vA * vB)
    }

    products.reduceByKey(_ + _)
  }

  // ---------------------------------------------------------------------------
  // 7) SpMM advanced: CSR (left) × CSC (right)
  //
  // A: CSRMatrix (row-compressed)
  // B: CSCMatrix (column-compressed)
  //
  // For every (row i, col j) pair, do a local merge-style dot product
  // over the two sorted index lists. This is a clean CSR–CSC kernel.
  // NOTE: cartesian may be expensive for very large #rows × #cols,
  // so this is best for smaller matrices or as an "advanced layout" demo.
  // ---------------------------------------------------------------------------
  def spmmCSRWithCSC(A: CSRMatrix, B: CSCMatrix): RDD[((Int, Int), Double)] = {
    // A in CSR: expand each row i → (k,(i,vA))
    val A_byK: RDD[(Int, (Int, Double))] = A.rows.flatMap { row =>
      val i = row.row
      val cols = row.colIdx
      val vals = row.values
      val out = new scala.collection.mutable.ArrayBuffer[(Int, (Int, Double))](cols.length)
      var t = 0
      while (t < cols.length) {
        out += ((cols(t), (i, vals(t))))
        t += 1
      }
      out
    }

    // B in CSC: expand each column j → (k,(j,vB))
    val B_byK: RDD[(Int, (Int, Double))] = B.cols.flatMap { col =>
      val j = col.col
      val rIdx = col.rowIdx
      val v    = col.values
      val out = new scala.collection.mutable.ArrayBuffer[(Int, (Int, Double))](rIdx.length)
      var t = 0
      while (t < rIdx.length) {
        out += ((rIdx(t), (j, v(t))))
        t += 1
      }
      out
    }

    // Co-partition to reduce shuffle, then join on k
    val (aPart, bPart) = coPartition(A_byK, B_byK)
    val joined = aPart.join(bPart) // (k, ((i,vA),(j,vB)))

    // Multiply and accumulate to C(i,j)
    val products = joined.map { case (_, ((i, vA), (j, vB))) => ((i, j), vA * vB) }
    products.reduceByKey(_ + _)

  }

  // ---------------------------------------------------------------------------
  // 8) SpMM: COO (left) x Dense (right)
  // ---------------------------------------------------------------------------
  def spmm_dense(A: SparseMatrix, B: DenseMatrix): RDD[(Int, Array[Double])] = {
    val AkeyedByJ: RDD[(Int, (Int, Double))] = A.entries.map { case (i, j, v) => (j, (i, v)) }
    val joined = AkeyedByJ.join(B.rows)  // <-- fix here

    val partials: RDD[(Int, Array[Double])] = joined.map { case (_, ((i, v), rowB)) =>
      val out = new Array[Double](rowB.length)
      var k = 0
      while (k < rowB.length) { out(k) = rowB(k) * v; k += 1 }
      (i, out)
    }

    partials.reduceByKey { (a, b) =>
      val len = math.max(a.length, b.length)
      val res = new Array[Double](len)
      var k = 0
      while (k < len) {
        val av = if (k < a.length) a(k) else 0.0
        val bv = if (k < b.length) b(k) else 0.0
        res(k) = av + bv
        k += 1
      }
      res
    }

  }

}

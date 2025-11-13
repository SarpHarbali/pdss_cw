package pdss.engine

import org.apache.spark.HashPartitioner
import org.apache.spark.rdd.RDD
import pdss.core._   // SparseMatrix, DistVector, CSRMatrix, CSRRow, CSCMatrix, CSCCol, DenseMatrix
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast

import scala.reflect.ClassTag

/**
 * Baseline + optimized execution engine built only on RDDs.
 * Allowed (per instructor):
 *  - COO × COO
 *  - CSR × CSR
 *  - CSC × CSC
 *
 * Not allowed:
 *  - mixing formats (CSR × COO, CSR × CSC, ...)
 */
object ExecutionEngine {

  // ---------------------------------------------------------------------------
  // helper: co-partition two RDDs on Int key
  // ---------------------------------------------------------------------------
  private def coPartition[V: ClassTag, W: ClassTag](
      left: RDD[(Int, V)],
      right: RDD[(Int, W)]
  ): (RDD[(Int, V)], RDD[(Int, W)]) = {
    val p = new HashPartitioner(16)
    val L = left.partitionBy(p).persist()
    val R = right.partitionBy(p).persist()
    (L, R)
  }
  private def requireMulCompat(
                                op: String,
                                aRows: Long, aCols: Long,
                                bRows: Long, bCols: Long
                              ): Unit = {
    require(
      aCols == bRows,
      s"Incompatible dimensions for $op: left is ${aRows}x${aCols}, right is ${bRows}x${bCols} (need left.nCols == right.nRows)"
    )
  }

  // ---------------------------------------------------------------------------
  // 1) SpMV: y = A * x  (COO × SparseVector)
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
  // ---------------------------------------------------------------------------
  def spmvCSR(A: CSRMatrix, x: DistVector): RDD[(Int, Double)] = {

    val AbyK: RDD[(Int, (Int, Double))] = A.rows.flatMap { row =>
      val i    = row.row
      val cols = row.colIdx
      val vals = row.values
      val buf  = new scala.collection.mutable.ArrayBuffer[(Int, (Int, Double))](cols.length)
      var t = 0
      while (t < cols.length) {
        buf += ((cols(t), (i, vals(t))))
        t += 1
      }
      buf
    }

    val xByJ: RDD[(Int, Double)] = x.values

    val (acp, xcp) = coPartition(AbyK, xByJ)

    val joined = acp.join(xcp)

    joined
      .map { case (_, ((i, vA), xj)) => (i, vA * xj) }
      .reduceByKey(_ + _)
  }

  // ---------------------------------------------------------------------------
  // 3) SpMV: y = A * x  (COO × DenseVector)
  // ---------------------------------------------------------------------------
  def spmvCooWithDense(A: SparseMatrix, x_bcast: Broadcast[Array[Double]]): RDD[(Int, Double)] = {
    val x = x_bcast.value
    A.entries
      .map { case (i, j, v) =>
        val xj = if (j < x.length) x(j) else 0.0
        (i, v * xj)
      }
      .reduceByKey(_ + _)
  }

  // ---------------------------------------------------------------------------
  // 4) SpMV advanced: y = A * x  (CSR × DenseVector)
  // ---------------------------------------------------------------------------
  def spmvCsrWithDense(A: CSRMatrix, x_bcast: Broadcast[Array[Double]]): RDD[(Int, Double)] = {
    val x = x_bcast.value
    A.rows.map { row =>
      val i    = row.row
      val cols = row.colIdx
      val vals = row.values

      var yi = 0.0
      var t  = 0
      while (t < cols.length) {
        val j = cols(t)
        if (j < x.length) {
          yi += vals(t) * x(j)
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
    requireMulCompat("spmmCSC (COO×COO)", A.nRows, A.nCols, B.nRows, B.nCols)

    val numParts =  16 // choose based on cluster & data size

    val AbyK = A.entries
      .map { case (i, k, vA) => (k, (i, vA)) }
      .partitionBy(new HashPartitioner(numParts))
      .persist()

    val BbyK = B.entries
      .map { case (k, j, vB) => (k, (j, vB)) }
      .partitionBy(AbyK.partitioner.get) // reuse
      .persist()
    val joined = AbyK.join(BbyK)  // Spark handles the shuffle

    val products = joined.map {
      case (_, ((i, vA), (j, vB))) =>
        ((i, j), vA * vB)
    }

    products.reduceByKey(_ + _)
  }

  // ---------------------------------------------------------------------------
  // 6) SpMM advanced: CSR × CSR
  //
  // expand A rows to (k, (i, vA))
  // expand B rows to (k, (j, vB))
  // join on k
  // ---------------------------------------------------------------------------
  def spmmCSR(A: CSRMatrix, B: CSRMatrix): RDD[((Int, Int), Double)] = {
    requireMulCompat("spmmCSC (CSR×CSR)", A.nRows, A.nCols, B.nRows, B.nCols)

    // A → (k, (i, vA))
    val A_byK: RDD[(Int, (Int, Double))] = A.rows.flatMap { row =>
      val i    = row.row
      val cols = row.colIdx
      val vals = row.values
      val out  = new scala.collection.mutable.ArrayBuffer[(Int, (Int, Double))](cols.length)
      var t = 0
      while (t < cols.length) {
        out += ((cols(t), (i, vals(t))))  // key by k
        t += 1
      }
      out
    }

    // B → (k, (j, vB))
    val B_byK: RDD[(Int, (Int, Double))] = B.rows.flatMap { row =>
      val k    = row.row
      val cols = row.colIdx
      val vals = row.values
      val out  = new scala.collection.mutable.ArrayBuffer[(Int, (Int, Double))](cols.length)
      var t = 0
      while (t < cols.length) {
        out += ((k, (cols(t), vals(t))))   // (k, (j, vB))
        t += 1
      }
      out
    }

    val (aPart, bPart) = coPartition(A_byK, B_byK)
    val joined = aPart.join(bPart) // (k, ((i,vA),(j,vB)))

    val products = joined.map {
      case (_, ((i, vA), (j, vB))) =>
        ((i, j), vA * vB)
    }

    products.reduceByKey(_ + _)
  }

  // ---------------------------------------------------------------------------
  // 7) SpMM advanced: CSC × CSC
  //
  // A: CSC → columns: for each col k, (row=i, vA)
  // B: CSC → columns: for each col j, (row=k, vB)
  // we join on k
  // ---------------------------------------------------------------------------
  def spmmCSC(A: CSCMatrix, B: CSCMatrix): RDD[((Int, Int), Double)] = {
    requireMulCompat("spmmCSC (CSC×CSC)", A.nRows, A.nCols, B.nRows, B.nCols)

    // A → (k, (i, vA))   where k = A.col
    val A_byK: RDD[(Int, (Int, Double))] = A.cols.flatMap { col =>
      val k    = col.col
      val rows = col.rowIdx
      val vals = col.values
      val out  = new scala.collection.mutable.ArrayBuffer[(Int, (Int, Double))](rows.length)
      var t = 0
      while (t < rows.length) {
        val i  = rows(t)
        val vA = vals(t)
        out += ((k, (i, vA)))
        t += 1
      }
      out
    }

    // B → (k, (j, vB))   where k = rowIdx of B's column j
    val B_byK: RDD[(Int, (Int, Double))] = B.cols.flatMap { col =>
      val j    = col.col
      val rows = col.rowIdx
      val vals = col.values
      val out  = new scala.collection.mutable.ArrayBuffer[(Int, (Int, Double))](rows.length)
      var t = 0
      while (t < rows.length) {
        val k  = rows(t)   // shared dim
        val vB = vals(t)
        out += ((k, (j, vB)))
        t += 1
      }
      out
    }

    val (aPart, bPart) = coPartition(A_byK, B_byK)
    val joined = aPart.join(bPart) // (k, ((i,vA),(j,vB)))

    val products = joined.map {
      case (_, ((i, vA), (j, vB))) =>
        ((i, j), vA * vB)
    }

    products.reduceByKey(_ + _)
  }

  // ---------------------------------------------------------------------------
  // 8) SpMM: COO (left) x Dense (right)
  // ---------------------------------------------------------------------------
  def spmm_dense(A: SparseMatrix, B: DenseMatrix): RDD[(Int, Array[Double])] = {
    requireMulCompat("spmmCSC (COO×Dense)", A.nRows, A.nCols, B.nRows, B.nCols)

    val AkeyedByJ: RDD[(Int, (Int, Double))] = A.entries.map { case (i, j, v) => (j, (i, v)) }
    val joined = AkeyedByJ.join(B.rows)

    val partials: RDD[(Int, Array[Double])] = joined.map {
      case (_, ((i, v), rowB)) =>
        val out = new Array[Double](rowB.length)
        var k = 0
        while (k < rowB.length) {
          out(k) = rowB(k) * v
          k += 1
        }
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


}

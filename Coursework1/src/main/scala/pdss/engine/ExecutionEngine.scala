package pdss.engine

import org.apache.spark.HashPartitioner
import org.apache.spark.rdd.RDD
import pdss.core._
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast

import scala.reflect.ClassTag


object ExecutionEngine {

  private def coPartition[V: ClassTag, W: ClassTag](
      left: RDD[(Int, V)],
      right: RDD[(Int, W)]
  ): (RDD[(Int, V)], RDD[(Int, W)]) = {
    val p = new HashPartitioner(left.sparkContext.defaultParallelism * 2)
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

  def spmv(A: SparseMatrix, x: DistVector): RDD[(Int, Double)] = {
    val AkeyedByJ: RDD[(Int, (Int, Double))] =
      A.entries.map { case (i, j, v) => (j, (i, v)) }

    val xByJ: RDD[(Int, Double)] = x.values

    val joined = AkeyedByJ.join(xByJ)

    joined
      .map { case (_, ((i, v), xj)) => (i, v * xj) }
      .reduceByKey(_ + _)
  }

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

  def spmvCooWithDense(A: SparseMatrix, x_bcast: Broadcast[Array[Double]]): RDD[(Int, Double)] = {
    val x = x_bcast.value
    A.entries
      .map { case (i, j, v) =>
        val xj = if (j < x.length) x(j) else 0.0
        (i, v * xj)
      }
      .reduceByKey(_ + _)
  }


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


  def spmm(A: SparseMatrix, B: SparseMatrix): RDD[((Int, Int), Double)] = {
    requireMulCompat("spmmCSC (COO×COO)", A.nRows, A.nCols, B.nRows, B.nCols)

    val numParts =  16

    val AbyK = A.entries
      .map { case (i, k, vA) => (k, (i, vA)) }
      .partitionBy(new HashPartitioner(numParts))
      .persist()

    val BbyK = B.entries
      .map { case (k, j, vB) => (k, (j, vB)) }
      .partitionBy(AbyK.partitioner.get)
      .persist()
    val joined = AbyK.join(BbyK)

    val products = joined.map {
      case (_, ((i, vA), (j, vB))) =>
        ((i, j), vA * vB)
    }

    products.reduceByKey(_ + _)
  }


  def spmmCSR(A: CSRMatrix, B: CSRMatrix): RDD[((Int, Int), Double)] = {
    requireMulCompat("spmmCSC (CSR×CSR)", A.nRows, A.nCols, B.nRows, B.nCols)

    val A_byK: RDD[(Int, (Int, Double))] = A.rows.flatMap { row =>
      val i    = row.row
      val cols = row.colIdx
      val vals = row.values
      val out  = new scala.collection.mutable.ArrayBuffer[(Int, (Int, Double))](cols.length)
      var t = 0
      while (t < cols.length) {
        out += ((cols(t), (i, vals(t))))
        t += 1
      }
      out
    }

    val B_byK: RDD[(Int, (Int, Double))] = B.rows.flatMap { row =>
      val k    = row.row
      val cols = row.colIdx
      val vals = row.values
      val out  = new scala.collection.mutable.ArrayBuffer[(Int, (Int, Double))](cols.length)
      var t = 0
      while (t < cols.length) {
        out += ((k, (cols(t), vals(t))))
        t += 1
      }
      out
    }

    val (aPart, bPart) = coPartition(A_byK, B_byK)
    val joined = aPart.join(bPart)

    val products = joined.map {
      case (_, ((i, vA), (j, vB))) =>
        ((i, j), vA * vB)
    }

    products.reduceByKey(_ + _)
  }


  def spmmCSC(A: CSCMatrix, B: CSCMatrix): RDD[((Int, Int), Double)] = {
    requireMulCompat("spmmCSC (CSC×CSC)", A.nRows, A.nCols, B.nRows, B.nCols)

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

    val B_byK: RDD[(Int, (Int, Double))] = B.cols.flatMap { col =>
      val j    = col.col
      val rows = col.rowIdx
      val vals = col.values
      val out  = new scala.collection.mutable.ArrayBuffer[(Int, (Int, Double))](rows.length)
      var t = 0
      while (t < rows.length) {
        val k  = rows(t)
        val vB = vals(t)
        out += ((k, (j, vB)))
        t += 1
      }
      out
    }

    val (aPart, bPart) = coPartition(A_byK, B_byK)
    val joined = aPart.join(bPart)

    val products = joined.map {
      case (_, ((i, vA), (j, vB))) =>
        ((i, j), vA * vB)
    }

    products.reduceByKey(_ + _)
  }

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

    val (aPart, bPart) = coPartition(A_byK, B_byK)
    val joined = aPart.join(bPart)

    val products = joined.map { case (_, ((i, vA), (j, vB))) => ((i, j), vA * vB) }
    products.reduceByKey(_ + _)

  }


}

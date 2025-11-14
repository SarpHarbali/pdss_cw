package pdss

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import pdss.core._
import pdss.io.Loader
import pdss.frontend.LinearAlgebraAPI
import pdss.frontend.LinearAlgebraAPI.SparseFormat._
import pdss.engine.ExecutionEngine

object FrontendTests {

  private def approxEq(a: Double, b: Double, eps: Double = 1e-6): Boolean =
    math.abs(a - b) <= eps

  private def maxDiffVectors(a: DistVector, b: DistVector): Double = {
    val diffs: RDD[Double] = a.values.fullOuterJoin(b.values).map {
      case (_, (vaOpt, vbOpt)) =>
        val va = vaOpt.getOrElse(0.0)
        val vb = vbOpt.getOrElse(0.0)
        math.abs(va - vb)
    }
    diffs.fold(0.0)((acc, x) => math.max(acc, x))
  }

  private def maxDiffSparseMatrices(a: SparseMatrix, b: SparseMatrix): Double = {
    val aByKey = a.entries.map { case (i, j, v) => ((i, j), v) }
    val bByKey = b.entries.map { case (i, j, v) => ((i, j), v) }
    val diffs: RDD[Double] = aByKey.fullOuterJoin(bByKey).map {
      case (_, (vaOpt, vbOpt)) =>
        val va = vaOpt.getOrElse(0.0)
        val vb = vbOpt.getOrElse(0.0)
        math.abs(va - vb)
    }
    diffs.fold(0.0)((acc, x) => math.max(acc, x))
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[*]").setAppName("FrontendTests")
    val sc   = new SparkContext(conf)
    try {
      testSpmvSparseVsDense(sc)
      testSpmmFormatsAgree(sc)
      testSpmmSparseDense(sc)
      testCsvLoaderSpmv(sc)
      testTensorMttkrpSimple(sc)
      println()
      println("All frontend tests passed.")
    } finally {
      sc.stop()
    }
  }

  def testSpmvSparseVsDense(sc: SparkContext): Unit = {
    println("Running testSpmvSparseVsDense...")

    val triples = sc.parallelize(Seq(
      (0, 0, 1.0),
      (0, 2, 2.0),
      (1, 1, 3.0),
      (2, 0, 4.0),
      (2, 1, 5.0)
    ))
    val A = SparseMatrix(triples, nRows = 3, nCols = 3)

    val xSparse = DistVector(sc.parallelize(Seq(
      (0, 2.0),
      (2, 1.0)
    )), length = 3)

    val xDense = Array(2.0, 0.0, 1.0)

    val yCooSparse = LinearAlgebraAPI
      .spmv(A, xSparse, useCSR = false)

    val yCsrSparse = LinearAlgebraAPI
      .spmv(A, xSparse, useCSR = true)

    val yCooDense = LinearAlgebraAPI
      .spmv(A, xDense, useCSR = false)(sc)

    val yCsrDense = LinearAlgebraAPI
      .spmv(A, xDense, useCSR = true)(sc)

    val d12 = maxDiffVectors(yCooSparse, yCsrSparse)
    val d13 = maxDiffVectors(yCooSparse, yCooDense)
    val d14 = maxDiffVectors(yCooSparse, yCsrDense)

    assert(d12 <= 1e-6, s"COO-sparse vs CSR-sparse mismatch, max diff = $d12")
    assert(d13 <= 1e-6, s"COO-sparse vs COO-dense mismatch, max diff = $d13")
    assert(d14 <= 1e-6, s"COO-sparse vs CSR-dense mismatch, max diff = $d14")

    println("testSpmvSparseVsDense passed.")
  }

  def testSpmmFormatsAgree(sc: SparkContext): Unit = {
    println("Running testSpmmFormatsAgree...")

    val triples = sc.parallelize(Seq(
      (0, 0, 1.0),
      (0, 2, 2.0),
      (1, 1, 3.0),
      (2, 0, 4.0),
      (2, 1, 5.0)
    ))
    val A = SparseMatrix(triples, nRows = 3, nCols = 3)
    val B = A

    val cooMat = LinearAlgebraAPI.spmm(A, B, COO)
    val csrMat = LinearAlgebraAPI.spmm(A, B, CSR)
    val cscMat = LinearAlgebraAPI.spmm(A, B, CSC)

    val csrA = Loader.cooToCSR(A)
    val cscB = Loader.cooToCSC(B)
    val csrCscPairs = ExecutionEngine.spmmCSRWithCSC(csrA, cscB)
    val csrCscMat = SparseMatrix(
      csrCscPairs.map { case ((i, j), v) => (i, j, v) },
      nRows = A.nRows,
      nCols = B.nCols
    )

    val dCooCsr = maxDiffSparseMatrices(cooMat, csrMat)
    val dCooCsc = maxDiffSparseMatrices(cooMat, cscMat)
    val dCooCsrCsc = maxDiffSparseMatrices(cooMat, csrCscMat)

    assert(dCooCsr <= 1e-6, s"COO vs CSR mismatch, max diff = $dCooCsr")
    assert(dCooCsc <= 1e-6, s"COO vs CSC mismatch, max diff = $dCooCsc")
    assert(dCooCsrCsc <= 1e-6, s"COO vs CSR×CSC mismatch, max diff = $dCooCsrCsc")

    println("testSpmmFormatsAgree passed.")
  }

  def testSpmmSparseDense(sc: SparkContext): Unit = {
    println("Running testSpmmSparseDense...")

    val A = SparseMatrix(sc.parallelize(Seq(
      (0, 0, 1.0),
      (0, 2, 2.0),
      (1, 1, 3.0),
      (2, 0, 4.0),
      (2, 1, 5.0)
    )), nRows = 3, nCols = 3)

    val BRows = Array(
      Array(1.0, 2.0),
      Array(0.0, 3.0),
      Array(4.0, 5.0)
    )

    val B_dense = DenseMatrix(
      sc.parallelize(Seq(
        (0, BRows(0)),
        (1, BRows(1)),
        (2, BRows(2))
      )),
      nRows = 3,
      nCols = 2
    )

    val C = LinearAlgebraAPI.spmm(A, B_dense)

    val A_local = Array(
      Array(1.0, 0.0, 2.0),
      Array(0.0, 3.0, 0.0),
      Array(4.0, 5.0, 0.0)
    )

    def mult(A: Array[Array[Double]], B: Array[Array[Double]]): Array[Array[Double]] = {
      val m = A.length
      val k = A(0).length
      val n = B(0).length
      val C = Array.ofDim[Double](m, n)
      var i = 0
      while (i < m) {
        var j = 0
        while (j < n) {
          var s = 0.0
          var t = 0
          while (t < k) {
            s += A(i)(t) * B(t)(j)
            t += 1
          }
          C(i)(j) = s
          j += 1
        }
        i += 1
      }
      C
    }

    val C_local = mult(A_local, BRows)
    val expectedBC = sc.broadcast(C_local)

    val rowMaxDiffs: RDD[Double] = C.rows.map {
      case (i, row) =>
        val expRow = expectedBC.value(i)
        var maxd = 0.0
        var j = 0
        while (j < row.length && j < expRow.length) {
          val d = math.abs(row(j) - expRow(j))
          if (d > maxd) maxd = d
          j += 1
        }
        maxd
    }

    val maxDiff = rowMaxDiffs.fold(0.0)((acc, x) => math.max(acc, x))
    assert(maxDiff <= 1e-6, s"Sparse×Dense SpMM mismatch, max diff = $maxDiff")

    println("testSpmmSparseDense passed.")
  }

  def testCsvLoaderSpmv(sc: SparkContext): Unit = {
    println("Running testCsvLoaderSpmv...")

    val matrixAPath   = "src/main/data/matrix_A.csv"
    val vectorCsvPath = "src/main/data/vector_x.csv"

    val A = Loader.loadCSVToCOO(sc, matrixAPath)
    val x = Loader.loadVector(sc, vectorCsvPath)

    val y = LinearAlgebraAPI.spmv(A, x, useCSR = false).values

    val count = y.count()
    assert(count > 0, "CSV-based SpMV produced empty result")

    val badCount = y.map { case (_, v) => v.isNaN || v.isInfinity }.filter(identity).count()
    assert(badCount == 0, s"CSV-based SpMV produced $badCount NaN/Inf values")

    println("testCsvLoaderSpmv passed.")
  }

  def testTensorMttkrpSimple(sc: SparkContext): Unit = {
    println("Running testTensorMttkrpSimple...")

    val entries: RDD[(Array[Int], Double)] = sc.parallelize(Seq(
      (Array(0, 0, 0), 1.0),
      (Array(0, 0, 1), 2.0),
      (Array(1, 1, 0), 3.0),
      (Array(1, 1, 1), 4.0)
    ))
    val tensor = SparseTensor(entries, shape = Array(2, 2, 2))

    def onesMat(): DenseMatrix = {
      val rows = sc.parallelize(Seq(
        (0, Array(1.0, 1.0)),
        (1, Array(1.0, 1.0))
      ))
      DenseMatrix(rows, nRows = 2, nCols = 2)
    }

    val U = onesMat()
    val V = onesMat()
    val W = onesMat()

    val M = LinearAlgebraAPI.mttkrp(tensor, Seq(U, V, W), targetMode = 0)

    val expected = Map(
      0 -> Array(3.0, 3.0),
      1 -> Array(7.0, 7.0)
    )
    val expectedBC = sc.broadcast(expected)

    val rowMaxDiffs: RDD[Double] = M.rows.map {
      case (i, row) =>
        val expRow = expectedBC.value(i)
        var maxd = 0.0
        var j = 0
        while (j < row.length && j < expRow.length) {
          val d = math.abs(row(j) - expRow(j))
          if (d > maxd) maxd = d
          j += 1
        }
        maxd
    }

    val maxDiff = rowMaxDiffs.fold(0.0)((acc, x) => math.max(acc, x))
    assert(maxDiff <= 1e-6, s"MTTKRP simple test mismatch, max diff = $maxDiff")

    println("testTensorMttkrpSimple passed.")
  }
}

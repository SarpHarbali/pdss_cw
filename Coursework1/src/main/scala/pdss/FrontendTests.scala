package pdss

import org.apache.spark.{SparkConf, SparkContext}
import pdss.core._
import pdss.io.Loader
import pdss.frontend.LinearAlgebraAPI
import pdss.frontend.LinearAlgebraAPI.SparseFormat._
import org.apache.spark.rdd.RDD

object FrontendTests {

  private def approxEq(a: Double, b: Double, eps: Double = 1e-6): Boolean =
    math.abs(a - b) <= eps

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[*]").setAppName("FrontendTests")
    val sc   = new SparkContext(conf)

    try {
      testSpmvSparseVsDense(sc)
      testSpmmFormatsAgree(sc)
      testSpmmSparseDense(sc)
      testCsvLoaderSpmv(sc)
      testTensorMttkrpSimple(sc)
      println("\n All frontend tests passed.")
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

    val y_coo_sparse = LinearAlgebraAPI
      .spmv(A, xSparse, useCSR = false)
      .values.collect().toMap

    val y_csr_sparse = LinearAlgebraAPI
      .spmv(A, xSparse, useCSR = true)
      .values.collect().toMap

    val y_coo_dense = LinearAlgebraAPI
      .spmv(A, xDense, useCSR = false)(sc)
      .values.collect().toMap

    val y_csr_dense = LinearAlgebraAPI
      .spmv(A, xDense, useCSR = true)(sc)
      .values.collect().toMap


    val allKeys =
      (y_coo_sparse.keySet ++ y_csr_sparse.keySet ++ y_coo_dense.keySet ++ y_csr_dense.keySet)

    def get(m: Map[Int, Double], i: Int): Double =
      m.getOrElse(i, 0.0)

    for (i <- allKeys) {
      val v1 = get(y_coo_sparse, i)
      val v2 = get(y_csr_sparse, i)
      val v3 = get(y_coo_dense, i)
      val v4 = get(y_csr_dense, i)

      assert(approxEq(v1, v2), s"COO-sparse vs CSR-sparse mismatch at row $i: $v1 vs $v2")
      assert(approxEq(v1, v3), s"COO-sparse vs COO-dense mismatch at row $i: $v1 vs $v3")
      assert(approxEq(v1, v4), s"COO-sparse vs CSR-dense mismatch at row $i: $v1 vs $v4")
    }

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

    def toMap(m: SparseMatrix): Map[(Int, Int), Double] =
      m.entries.collect().map { case (i, j, v) => ((i, j), v) }.toMap

    val cooRes = toMap(LinearAlgebraAPI.spmm(A, B, COO))
    val csrRes = toMap(LinearAlgebraAPI.spmm(A, B, CSR))
    val cscRes = toMap(LinearAlgebraAPI.spmm(A, B, CSC))

    assert(cooRes.keySet == csrRes.keySet)
    assert(cooRes.keySet == cscRes.keySet)

    for (k <- cooRes.keys) {
      assert(approxEq(cooRes(k), csrRes(k)))
      assert(approxEq(cooRes(k), cscRes(k)))
    }

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

    val B_rows = Array(
      Array(1.0, 2.0),
      Array(0.0, 3.0),
      Array(4.0, 5.0)
    )

    val B_dense = DenseMatrix(
      sc.parallelize(Seq(
        (0, B_rows(0)),
        (1, B_rows(1)),
        (2, B_rows(2))
      )),
      nRows = 3,
      nCols = 2
    )

    val C = LinearAlgebraAPI.spmm(A, B_dense)
    val C_map = C.rows.collect().toMap

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

    val C_local = mult(A_local, B_rows)

    for (i <- 0 until 3) {
      val row = C_map(i)
      for (j <- row.indices) {
        assert(approxEq(row(j), C_local(i)(j)))
      }
    }

    println("testSpmmSparseDense passed.")
  }

  def testCsvLoaderSpmv(sc: SparkContext): Unit = {
    println("Running testCsvLoaderSpmv...")

    val matrixAPath  = "src/main/data/matrix_A.csv"
    val vectorCsvPath = "src/main/data/vector_x.csv"

    val A = Loader.loadCSVToCOO(sc, matrixAPath)
    val x = Loader.loadVector(sc, vectorCsvPath)

    val y = LinearAlgebraAPI.spmv(A, x, useCSR = false)
      .values.collect()

    assert(y.nonEmpty)
    assert(y.forall { case (_, v) => !v.isNaN && !v.isInfinity })

    println("testCsvLoaderSpmv passed.")
  }


  def testTensorMttkrpSimple(sc: SparkContext): Unit = {
    println("Running testTensorMttkrpSimple...")

    val entries: RDD[(Array[Int], Double)] = sc.parallelize(Seq(
      (Array(0,0,0), 1.0),
      (Array(0,0,1), 2.0),
      (Array(1,1,0), 3.0),
      (Array(1,1,1), 4.0)
    ))
    val tensor = SparseTensor(entries, shape = Array(2,2,2))

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
    val M_rows = M.rows.collect().toMap

    val expected = Map(
      0 -> Array(3.0, 3.0),
      1 -> Array(7.0, 7.0)
    )

    for ((i, row) <- M_rows) {
      val exp = expected(i)
      for (j <- row.indices) {
        assert(approxEq(row(j), exp(j)))
      }
    }

    println("testTensorMttkrpSimple passed.")
  }
}

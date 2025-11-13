package pdss

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{Matrices, DenseMatrix => MLDenseMatrix}
import pdss.io.Loader
import pdss.engine.ExecutionEngine
import pdss.core.{DenseMatrix => CoreDenseMatrix}

object MLLib {

  // Convert your PDSS DenseMatrix → MLlib DenseMatrix
  private def pdssDenseToMLlib(m: CoreDenseMatrix): MLDenseMatrix = {
    val numRows = m.nRows.toInt
    val numCols = m.nCols.toInt

    // MLlib is column-major
    val data = Array.ofDim[Double](numRows * numCols)

    // m.rows: RDD[(Int, Array[Double])]
    m.rows.collect().foreach { case (i, row) =>
      var j = 0
      while (j < numCols) {
        data(j * numRows + i) = row(j) // column-major layout
        j += 1
      }
    }

    // Matrices.dense returns a Matrix which is actually a DenseMatrix
    Matrices.dense(numRows, numCols, data).asInstanceOf[MLDenseMatrix]
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("SimpleSpMMCompare")
      .setMaster("local[*]")
      .set("spark.ui.enabled", "false")

    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    // ------------------------------------------------
    // 1) OUR IMPLEMENTATION (COO × COO)
    // ------------------------------------------------
    val A = Loader.loadCSVToCOO(sc, "src/main/data/sparse_matrix_normal.csv")
    val B = Loader.loadCSVToCOO(sc, "src/main/data/sparse_matrix_normal2.csv")

    require(A.nCols == B.nRows)

    // Warm up
    ExecutionEngine.spmm(A, B).count()

    val startOurs = System.currentTimeMillis()
    ExecutionEngine.spmm(A, B).count()
    val endOurs = System.currentTimeMillis()
    val oursTime = endOurs - startOurs

    println(s"Our SpMM time: $oursTime ms")

    // ------------------------------------------------
    // 2) MLLIB LOCAL DenseMatrix × DenseMatrix
    // ------------------------------------------------
    val A_dense: CoreDenseMatrix = Loader.loadDenseMatrixRows(sc, "src/main/data/sparse_matrix_normal.csv")
    val B_dense: CoreDenseMatrix = Loader.loadDenseMatrixRows(sc, "src/main/data/sparse_matrix_normal2.csv")

    require(A_dense.nCols == B_dense.nRows)

    val A_local: MLDenseMatrix = pdssDenseToMLlib(A_dense)
    val B_local: MLDenseMatrix = pdssDenseToMLlib(B_dense)

    // Warm up
    A_local.multiply(B_local)

    val startMLlib = System.currentTimeMillis()
    val C_local: MLDenseMatrix = A_local.multiply(B_local)
    val endMLlib = System.currentTimeMillis()
    val mllibTime = endMLlib - startMLlib

    println(s"MLlib local DenseMatrix.multiply time: $mllibTime ms")
    println(s"C dims: ${C_local.numRows} x ${C_local.numCols}")

    // ------------------------------------------------
    // 3) SPEEDUP
    // ------------------------------------------------
    val speedup = mllibTime.toDouble / oursTime.toDouble
    println(f"Speedup (MLlibTime / OurTime): $speedup%.2fx")

    sc.stop()
  }
}

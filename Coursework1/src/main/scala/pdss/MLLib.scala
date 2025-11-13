package pdss

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.linalg.{Matrices, DenseMatrix => MLDenseMatrix}
import pdss.io.Loader
import pdss.engine.ExecutionEngine
import pdss.core.{DenseMatrix => CoreDenseMatrix}
import org.apache.spark.mllib.linalg.distributed.{IndexedRowMatrix, IndexedRow}
import pdss.core._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{IndexedRowMatrix, IndexedRow}



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

  private def coordinateToLocalDense(m: SparseMatrix): org.apache.spark.mllib.linalg.Matrix = {
    val numRows = m.nRows.toInt
    val numCols = m.nCols.toInt

    // Column-major layout for MLlib
    val data = Array.ofDim[Double](numRows * numCols)

    // m.entries is RDD[(i, j, v)] in COO format
    m.entries.collect().foreach { case (i, j, v) =>
      // MLlib DenseMatrix is column-major: index = j * numRows + i
      data(j * numRows + i) = v
    }

    org.apache.spark.mllib.linalg.Matrices.dense(numRows, numCols, data)
  }


  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("SpMM-MLLib-SizeBenchmark")
      .setMaster("local[*]")
      .set("spark.ui.enabled", "false")

    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    // Where your CSV input matrices live
    val baseDir = "src/main/data"

    // Matrix sizes to test
    val sizes = Seq(2500, 5000, 7500, 10000)

    import java.io.{File, PrintWriter}
    val out = new PrintWriter(new File("mllib_vs_pdss_sizes.csv"))
    out.println("size,ours_sec,mllib_sec,speedup_mllib_over_ours")

    sizes.foreach { n =>
      println()
      println(s"===================== n = $n =====================")

      val pathA = s"$baseDir/sparse_matrix_${n}_A.csv"
      val pathB = s"$baseDir/sparse_matrix_${n}_B.csv"

      // ------------------------------------------------
      // 1) OUR IMPLEMENTATION (CSR × CSC)
      // ------------------------------------------------
      val A = Loader.loadCSVToCOO(sc, pathA)
      val B = Loader.loadCSVToCOO(sc, pathB)

      val A_csr = Loader.cooToCSR(A)
      val B_csc = Loader.cooToCSC(B)

      require(A.nCols == B.nRows, s"Incompatible shapes for n=$n")

      // Warm up our engine
      ExecutionEngine.spmmCSRWithCSC(A_csr, B_csc).count()

      val startOurs = System.nanoTime()
      ExecutionEngine.spmmCSRWithCSC(A_csr, B_csc).count()
      val endOurs   = System.nanoTime()
      val oursTimeSec = (endOurs - startOurs) / 1e9

      println(f"Our SpMM time (n=$n): $oursTimeSec%.6f s")

      // ------------------------------------------------
      // 2) MLLIB: IndexedRowMatrix × local DenseMatrix
      // ------------------------------------------------
      val aRows = A.entries
        .groupBy(_._1)  // group by row index i
        .map { case (rowIdx, entries) =>
          val cols   = entries.map(_._2).toArray
          val values = entries.map(_._3).toArray
          IndexedRow(rowIdx.toLong, Vectors.sparse(A.nCols.toInt, cols, values))
        }

      val aMatrix = new IndexedRowMatrix(aRows, A.nRows, A.nCols.toInt)

      // B as a local dense MLlib matrix
      val bLocal = coordinateToLocalDense(B)

      // Warmup MLlib
      val warmupStart = System.nanoTime()
      val warmupResult = aMatrix.multiply(bLocal)
      warmupResult.rows.count()
      val warmupEnd = System.nanoTime()
      val warmupTimeSec = (warmupEnd - warmupStart) / 1e9
      println(f"MLlib warmup (n=$n): $warmupTimeSec%.6f s")

      // Actual MLlib timing
      val computeStart = System.nanoTime()
      val resultMatrix = aMatrix.multiply(bLocal)
      resultMatrix.rows.count()
      val computeEnd = System.nanoTime()
      val mllibTimeSec = (computeEnd - computeStart) / 1e9

      println(f"MLlib computation (n=$n): $mllibTimeSec%.6f s")

      // ------------------------------------------------
      // 3) SPEEDUP
      // ------------------------------------------------
      val speedup = mllibTimeSec / oursTimeSec
      println(f"Speedup (MLlib / Ours, n=$n): $speedup%.3fx")

      // Write a CSV row
      out.println(s"$n,$oursTimeSec,$mllibTimeSec,$speedup")
    }

    out.close()
    sc.stop()
  }








  //
//
//    val A_dense: CoreDenseMatrix = Loader.loadDenseMatrixRows(sc, "src/main/data/sparse_matrix_normal.csv")
//    val B_dense: CoreDenseMatrix = Loader.loadDenseMatrixRows(sc, "src/main/data/sparse_matrix_normal2.csv")
//
//    require(A_dense.nCols == B_dense.nRows)
//
//    val A_local: MLDenseMatrix = pdssDenseToMLlib(A_dense)
//    val B_local: MLDenseMatrix = pdssDenseToMLlib(B_dense)
//
//    // Warm up
//    A_local.multiply(B_local)
//
//    val startMLlib = System.currentTimeMillis()
//    A_local.multiply(B_local)
//    val endMLlib = System.currentTimeMillis()
//    val mllibTime = endMLlib - startMLlib
//
//    println(s"MLlib local DenseMatrix.multiply time: $mllibTime ms")




}

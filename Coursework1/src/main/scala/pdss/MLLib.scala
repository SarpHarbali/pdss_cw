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



  private def coordinateToLocalDense(m: SparseMatrix): org.apache.spark.mllib.linalg.Matrix = {
    val numRows = m.nRows.toInt
    val numCols = m.nCols.toInt

    val data = Array.ofDim[Double](numRows * numCols)

    m.entries.collect().foreach { case (i, j, v) =>
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

    val baseDir = "src/main/data"

    val sizes = Seq(2500, 5000, 7500, 10000)

    import java.io.{File, PrintWriter}
    val out = new PrintWriter(new File("mllib_vs_pdss_sizes.csv"))
    out.println("size,ours_sec,mllib_sec,speedup_mllib_over_ours")

    sizes.foreach { n =>
      println()
      println(s"===================== n = $n =====================")

      val pathA = s"$baseDir/sparse_matrix_${n}_A.csv"
      val pathB = s"$baseDir/sparse_matrix_${n}_B.csv"

      val A = Loader.loadCSVToCOO(sc, pathA)
      val B = Loader.loadCSVToCOO(sc, pathB)

      val A_csr = Loader.cooToCSR(A)
      val B_csc = Loader.cooToCSC(B)

      require(A.nCols == B.nRows, s"Incompatible shapes for n=$n")

      ExecutionEngine.spmmCSRWithCSC(A_csr, B_csc).count()

      val startOurs = System.nanoTime()
      ExecutionEngine.spmmCSRWithCSC(A_csr, B_csc).count()
      val endOurs   = System.nanoTime()
      val oursTimeSec = (endOurs - startOurs) / 1e9

      println(f"Our SpMM time (n=$n): $oursTimeSec%.6f s")


      val aRows = A.entries
        .groupBy(_._1)
        .map { case (rowIdx, entries) =>
          val cols   = entries.map(_._2).toArray
          val values = entries.map(_._3).toArray
          IndexedRow(rowIdx.toLong, Vectors.sparse(A.nCols.toInt, cols, values))
        }

      val aMatrix = new IndexedRowMatrix(aRows, A.nRows, A.nCols.toInt)

      val bLocal = coordinateToLocalDense(B)


      val warmupResult = aMatrix.multiply(bLocal)
      warmupResult.rows.count()

      val computeStart = System.nanoTime()
      val resultMatrix = aMatrix.multiply(bLocal)
      resultMatrix.rows.count()
      val computeEnd = System.nanoTime()
      val mllibTimeSec = (computeEnd - computeStart) / 1e9

      println(f"MLlib computation (n=$n): $mllibTimeSec%.6f s")

      val speedup = mllibTimeSec / oursTimeSec
      println(f"Speedup (MLlib / Ours, n=$n): $speedup%.3fx")

      out.println(s"$n,$oursTimeSec,$mllibTimeSec,$speedup")
    }

    out.close()
    sc.stop()
  }





}

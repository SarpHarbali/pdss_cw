package pdss

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.hadoop.fs.{FileSystem, Path}
import pdss.engine.ExecutionEngine
import pdss.engine.ChainPlanner
import pdss.io.Loader
import pdss.core._
import pdss.frontend.LinearAlgebraAPI
import pdss.frontend.LinearAlgebraAPI.SparseFormat._


object Main extends App {

  private def ensureOutputPathClear(sc: SparkContext, path: String): Unit = {
    val outPath = new Path(path)
    val fs = FileSystem.get(sc.hadoopConfiguration)
    if (fs.exists(outPath)) {
      fs.delete(outPath, true)
      println(s"Deleted existing output directory: $path")
    }
  }

  val conf = new SparkConf()
    .setAppName("PDSS-SpMM-Benchmark")
    .setMaster("local[*]")
    .set("spark.ui.enabled", "false")



  val sc = new SparkContext(conf)

  sc.setLogLevel("WARN")

  val A = Loader.loadCSVToCOO(sc, "data/sparse_matrix_normal.csv")
  val B = Loader.loadCSVToCOO(sc, "data/sparse_matrix_normal2.csv")
  val C = Loader.loadCSVToCOO(sc, "data/sparse_matrix_normal3.csv")

  val result = LinearAlgebraAPI.spmmChain3(A, B,C, format = COO)

  ensureOutputPathClear(sc, "results/test_output")


  result.entries
    .map { case (i, j, v) => s"$i,$j,$v" }
    .saveAsTextFile("results/test_output")


  sc.stop()
}

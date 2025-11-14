package pdss

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.hadoop.fs.{FileSystem, Path}
import pdss.engine.ExecutionEngine
import pdss.engine.ChainPlanner
import pdss.io.Loader
import pdss.core._

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

  val A = Loader.loadCSVToCOO(sc, "src/main/data/sparse_matrix_normal.csv")
  val B = Loader.loadCSVToCOO(sc, "src/main/data/sparse_matrix_normal2.csv")
  val C = Loader.loadCSVToCOO(sc, "src/main/data/sparse_matrix_normal3.csv")

  val plan = ChainPlanner.chooseOrder3(A, B, C)
  println(s"Chosen plan: ${plan.order}")
  println(f"Estimated first cost = ${plan.firstCost}%.2e, second = ${plan.secondCost}%.2e")

  def timePlan(label: String)(f: => pdss.core.SparseMatrix): Unit = {
    val start = System.nanoTime()
    val out = f
    val nnz = out.entries.count()
    val end = System.nanoTime()
    println(f"Elapsed for $label: ${(end - start)/1e9}%.3f sec | nnz=$nnz")
  }


  timePlan("chosen order") {
    if (plan.order == "AB_then_C") {
      println(f"AB then C")
      val AB  = ExecutionEngine.spmm(A, B).map { case ((i,j),v) => (i,j,v) }
      val ABm = SparseMatrix(AB, A.nRows, B.nCols)
      val ABC = ExecutionEngine.spmm(ABm, C).map { case ((i,j),v) => (i,j,v) }
      SparseMatrix(ABC, A.nRows, C.nCols)
    } else {
      val BC  = ExecutionEngine.spmm(B, C).map { case ((i,j),v) => (i,j,v) }
      val BCm = SparseMatrix(BC, B.nRows, C.nCols)
      val ABC = ExecutionEngine.spmm(A, BCm).map { case ((i,j),v) => (i,j,v) }
      SparseMatrix(ABC, A.nRows, C.nCols)
    }
  }

  timePlan("non-optimal order") {
    if (plan.order == "AB_then_C") {
      val BC  = ExecutionEngine.spmm(B, C).map { case ((i,j),v) => (i,j,v) }
      val BCm = SparseMatrix(BC, B.nRows, C.nCols)
      val ABC = ExecutionEngine.spmm(A, BCm).map { case ((i,j),v) => (i,j,v) }
      SparseMatrix(ABC, A.nRows, C.nCols)
    } else {
      val AB  = ExecutionEngine.spmm(A, B).map { case ((i,j),v) => (i,j,v) }
      val ABm = SparseMatrix(AB, A.nRows, B.nCols)
      val ABC = ExecutionEngine.spmm(ABm, C).map { case ((i,j),v) => (i,j,v) }
      SparseMatrix(ABC, A.nRows, C.nCols)
    }
  }

  sc.stop()
}

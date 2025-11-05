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

  // load as COO
  val A = Loader.loadCSVToCOO(sc, "src/main/data/sparse_matrix_normal.csv")
  val B = Loader.loadCSVToCOO(sc, "src/main/data/sparse_matrix_normal2.csv")
  val C = Loader.loadCSVToCOO(sc, "src/main/data/sparse_matrix_normal2.csv")

  val plan = ChainPlanner.chooseOrder3(A, B, C)
  println(s"Chosen plan: ${plan.order}")
  println(f"Estimated first cost = ${plan.firstCost}%.2e, second = ${plan.secondCost}%.2e")

  // --- Helper timers ---
  def timePlan(label: String)(f: => pdss.core.SparseMatrix): Unit = {
    val start = System.nanoTime()
    val out = f
    // ACTION: force evaluation & measure real work
    val nnz = out.entries.count()
    val end = System.nanoTime()
    println(f"Elapsed for $label: ${(end - start)/1e9}%.3f sec | nnz=$nnz")
  }


  // --- Execute the correct order ---
  timePlan(s"${plan.order} (chosen)") {
    if (plan.order == "AB_then_C") {
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
/*
  timePlan("Opposite (non-optimal) order") {
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
  }*/



















  /*
   // convert
   val Acsr = Loader.cooToCSR(Acoo)
   val Bcsc = Loader.cooToCSC(Bcoo)

   // 1) COO × COO
   println("▶ COO × COO")
   val t1 = System.nanoTime()
   val C_coo = ExecutionEngine.spmm(Acoo, Bcoo)
   C_coo.count()
   val t2 = System.nanoTime()
   val timeCoo = (t2 - t1) / 1e6
   println(f"✅ COO × COO took: $timeCoo%.2f ms")

   // 2) CSR × COO
   println("\n▶ CSR × COO")
   val t3 = System.nanoTime()
   val C_csr = ExecutionEngine.spmmCSRWithCOO(Acsr, Bcoo)
   C_csr.count()
   val t4 = System.nanoTime()
   val timeCsr = (t4 - t3) / 1e6
   println(f"✅ CSR × COO took: $timeCsr%.2f ms")

   // 3) CSR × CSC
   println("\n▶ CSR × CSC")
   val t5 = System.nanoTime()
   val C_csrcsc = ExecutionEngine.spmmCSRWithCSC(Acsr, Bcsc)
   C_csrcsc.count()
   val t6 = System.nanoTime()
   val timeCsrcsc = (t6 - t5) / 1e6
   println(f"✅ CSR × CSC took: $timeCsrcsc%.2f ms")

   // save
   ensureOutputPathClear(sc, "results/spmm_coo_output")
   ensureOutputPathClear(sc, "results/spmm_csr_output")
   ensureOutputPathClear(sc, "results/spmm_csrcsc_output")

   C_coo.map { case ((i,j),v) => s"$i,$j,$v" }.saveAsTextFile("results/spmm_coo_output")
   C_csr.map { case ((i,j),v) => s"$i,$j,$v" }.saveAsTextFile("results/spmm_csr_output")
   C_csrcsc.map { case ((i,j),v) => s"$i,$j,$v" }.saveAsTextFile("results/spmm_csrcsc_output")

   println(
     f"""
        |✅ Results written.
        |⏱ Summary:
        |  COO×COO    = $timeCoo%.2f ms
        |  CSR×COO    = $timeCsr%.2f ms
        |  CSR×CSC    = $timeCsrcsc%.2f ms
        |""".stripMargin
   )
 */
  sc.stop()
}

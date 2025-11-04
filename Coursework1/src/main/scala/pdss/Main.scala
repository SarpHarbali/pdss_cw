package pdss

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.hadoop.fs.{FileSystem, Path}
import pdss.engine.ExecutionEngine
import pdss.io.Loader

object Main extends App {

  def ensureOutputPathClear(sc: SparkContext, path: String): Unit = {
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
  val Acoo = Loader.loadCSVToCOO(sc, "src/main/data/sparse_matrix_normal.csv")
  val Bcoo = Loader.loadCSVToCOO(sc, "src/main/data/sparse_matrix_normal2.csv")

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

  sc.stop()
}

package pdss

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.hadoop.fs.{FileSystem, Path}
import pdss.engine.ExecutionEngine
import pdss.io.Loader

object Main extends App {
  /** Helper to ensure output directory doesn't exist before writing */
  private def ensureOutputPathClear(sc: SparkContext, path: String): Unit = {
    val outPath = new Path(path)
    val fs = FileSystem.get(sc.hadoopConfiguration)
    if (fs.exists(outPath)) {
      fs.delete(outPath, true) // recursive delete
      println(s"Deleted existing output directory: $path")
    }
  }
  val conf: SparkConf = new SparkConf()
    .setAppName("PDSS-SpMM-Benchmark")
    .setMaster("local[*]")
    .set("spark.ui.enabled", "false")

  val sc: SparkContext = new SparkContext(conf)

  // --- Load matrices in COO format ---
  val Acoo = Loader.loadCSVToCOO(sc, "src/main/data/sparse_matrix_normal.csv")
  val Bcoo = Loader.loadCSVToCOO(sc, "src/main/data/sparse_matrix_normal2.csv")

  // --- Convert A to CSR ---
  val Acsr = Loader.cooToCSR(Acoo)

  // ---------------- COO × COO ----------------
  println("▶ Running COO × COO SpMM ...")
  val t1 = System.nanoTime()
  val C_coo = ExecutionEngine.spmm(Acoo, Bcoo)

  // force full computation
  C_coo.count()
  val t2 = System.nanoTime()
  val timeCooMs = (t2 - t1) / 1e6
  println(f"✅ COO × COO SpMM took: $timeCooMs%.2f ms")

  // show first 10 just for inspection
  println("COO × COO first 10 results:")
  C_coo.take(10).foreach { case ((i, j), v) =>
    println(s"($i,$j) -> $v")
  }

  // ---------------- CSR × COO ----------------
  println("\n▶ Running CSR × COO SpMM ...")
  val t3 = System.nanoTime()
  val C_csr = ExecutionEngine.spmmCSRWithCOO(Acsr, Bcoo)

  // force full computation
  C_csr.count()
  val t4 = System.nanoTime()
  val timeCsrMs = (t4 - t3) / 1e6
  println(f"✅ CSR × COO SpMM took: $timeCsrMs%.2f ms")

  // show first 10 just for inspection
  println("CSR × COO first 10 results:")
  C_csr.take(10).foreach { case ((i, j), v) =>
    println(s"($i,$j) -> $v")
  }

  // --- Save both full results ---
  val cooOutputPath = "results/spmm_coo_output"
  val csrOutputPath = "results/spmm_csr_output"

  // Clear existing output directories if they exist
  ensureOutputPathClear(sc, cooOutputPath)
  ensureOutputPathClear(sc, csrOutputPath)

  // Save results
  C_coo.map { case ((i, j), v) => s"$i,$j,$v" }
    .saveAsTextFile(cooOutputPath)

  C_csr.map { case ((i, j), v) => s"$i,$j,$v" }
    .saveAsTextFile(csrOutputPath)

  println(
    s"\n✅ Results written to results/spmm_coo_output and results/spmm_csr_output\n" +
      f"⏱ Total Time Summary: COO×COO = $timeCooMs%.2f ms, CSR×COO = $timeCsrMs%.2f ms"
  )

  sc.stop()
}
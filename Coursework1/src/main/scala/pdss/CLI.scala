package pdss

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.storage.StorageLevel
import scala.io.StdIn.readLine

import pdss.core._
import pdss.engine.{ExecutionEngine, ChainPlanner}
import pdss.io.Loader

object CLI extends App {

  // ---------------- Spark ----------------
  val conf = new SparkConf()
    .setAppName("PDSS-CLI")
    .setMaster("local[*]")
    .set("spark.ui.enabled", "false")
  val sc = new SparkContext(conf)
  sc.setLogLevel("WARN")

  // ---------------- Helpers ----------------
  def prompt(s: String): String = { print(s); readLine().trim }

  def autoLoadSparse(sc: SparkContext, path: String): SparseMatrix = {
    val isProbablyCOO =
      path.toLowerCase.contains("coo") || path.toLowerCase.endsWith(".coo") || path.toLowerCase.contains("triplet")
    if (isProbablyCOO) Loader.loadCOO(sc, path) else Loader.loadCSVToCOO(sc, path)
  }

  def multiplySparse(A: SparseMatrix, B: SparseMatrix): SparseMatrix = {
    val Cij = ExecutionEngine.spmm(A, B)                 // ((i,j),v)
    val Ccoo = Cij.map { case ((i,j),v) => (i,j,v) }
    Ccoo.saveAsTextFile("results/new_output")
    SparseMatrix(Ccoo, A.nRows, B.nCols)
  }

  def timeAndCount(label: String, out: SparseMatrix): Unit = {
    val t0 = System.nanoTime()
    val nnz = out.entries.count()                        // force compute
    val t1 = System.nanoTime()
    println(f"$label => nnz=$nnz, time=${(t1 - t0)/1e9}%.3f s, dims=(${out.nRows} x ${out.nCols})")
  }

  def timeAndCountDense(label: String, out: DenseMatrix): Unit = {
    val t0 = System.nanoTime()
    val rows = out.rows.count()                          // force compute
    val t1 = System.nanoTime()
    println(f"$label => rows=$rows, time=${(t1 - t0)/1e9}%.3f s, dims=(${out.nRows} x ${out.nCols})")
  }

  // ---------------- Menu ----------------
  println(
    """PDSS CLI
      |Choose operation:
      |  1) SparseMatrix × SparseMatrix (collect chain, then pick order)
      |  2) SparseMatrix × DenseMatrix  (single step)
      |  q) Quit
      |""".stripMargin)

  prompt("Your choice (1/2/q): ").toLowerCase match {

    // ===== Mode 1: Sparse × Sparse (CHAIN) =====
    case "1" =>
      import scala.collection.mutable.ArrayBuffer
      val mats = ArrayBuffer.empty[SparseMatrix]

      // Collect ALL inputs first (no multiplication yet)
      var keep = true
      while (keep) {
        val p = prompt("Path to SparseMatrix CSV (COO i,j,v OR dense-like grid): ")
        val M = autoLoadSparse(sc, p)
        // cache input once to avoid rereads during chain
        M.entries.persist(StorageLevel.MEMORY_ONLY); M.entries.count()
        mats += M
        keep = prompt("Add another RIGHT operand? (y/n): ").equalsIgnoreCase("y")
      }

      if (mats.size < 2) {
        println("Need at least two matrices. Exiting."); sc.stop(); sys.exit(0)
      }

      val result: SparseMatrix =
        if (mats.size == 2) {
          println("Order: (A·B).")
          multiplySparse(mats(0), mats(1))
        } else if (mats.size == 3) {
          // Use YOUR planner for A,B,C
          val plan = ChainPlanner.chooseOrder3(mats(0), mats(1), mats(2))
          println(s"Planner chose: ${plan.order}  (est first=${"%.2e".format(plan.firstCost)}, second=${"%.2e".format(plan.secondCost)})")

          if (plan.order == "AB_then_C") {
            val AB = multiplySparse(mats(0), mats(1))
            multiplySparse(AB, mats(2))
          } else {
            val BC = multiplySparse(mats(1), mats(2))
            multiplySparse(mats(0), BC)
          }
        } else {
          // For 4+ let user pick a simple global strategy
          println(s"Chain of ${mats.size} matrices.")
          println("Choose global order:")
          println("  1) Left-deep  ((((A·B)·C)·D)·...)")
          println("  2) Right-deep (A·(B·(C·(D·...))))")
          val ord = prompt("Your choice (1/2): ").trim
          ord match {
            case "2" => mats.reduceRight { (L, R) => multiplySparse(L, R) }
            case _   => mats.reduce       { (acc, next) => multiplySparse(acc, next) }
          }
        }

      timeAndCount("Result (Sparse×Sparse chain)", result)

    // ===== Mode 2: Sparse × Dense (SINGLE) =====
    case "2" =>
      val aPath = prompt("Path to LEFT SparseMatrix CSV (COO or dense-like): ")
      val A = autoLoadSparse(sc, aPath)

      val bPath = prompt("Path to RIGHT DenseMatrix CSV (each line: v1,v2,...): ")
      val B = Loader.loadDenseMatrixRows(sc, bPath)

      val rows = ExecutionEngine.spmm_dense(A, B)        // (i, Array[Double])
      val out  = DenseMatrix(rows, A.nRows, B.nCols)
      timeAndCountDense("Result (Sparse×Dense)", out)

    case "q" | "quit" =>
      println("Bye."); sc.stop(); sys.exit(0)

    case other =>
      println(s"Unknown option: $other"); sc.stop(); sys.exit(1)
  }

  sc.stop()
}

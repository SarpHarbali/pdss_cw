package pdss

import org.apache.spark.{SparkConf, SparkContext}
import pdss.engine.{ExecutionEngine, ChainPlanner}
import pdss.io.Loader
import pdss.core._
import java.io.{File, PrintWriter}

object ChainOrderBenchmark {

  private def nowMs: Long = System.nanoTime() / 1000000L
  private def dstr(d: Double): String = f"$d%.1f"

  def main(args: Array[String]): Unit = {
    val threads   = 8
    val densities = Seq(0.1, 0.2, 0.3, 0.4)
    val outFile   = new File("spmm_chain_order_results.csv")
    val writer    = new PrintWriter(outFile)

    writer.println("density,plan,order,mA,nA,mB,nB,mC,nC,outNNZ,elapsed_ms,est_first,est_second,speedup_vs_alt")

    val conf = new SparkConf()
      .setAppName("PDSS-ChainOrder-Benchmark")
      .setMaster(s"local[$threads]")
      .set("spark.ui.enabled", "false")

    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    def runAB_then_C(A: SparseMatrix, B: SparseMatrix, C: SparseMatrix): (Long, Long) = {
      val t0  = nowMs
      val AB  = ExecutionEngine.spmm(A, B).map { case ((i, j), v) => (i, j, v) }
      val ABm = SparseMatrix(AB, A.nRows, B.nCols)
      val ABC = ExecutionEngine.spmm(ABm, C).map { case ((i, j), v) => (i, j, v) }
      val out = SparseMatrix(ABC, A.nRows, C.nCols)
      val nnz = out.entries.count()
      val t1  = nowMs
      (nnz, t1 - t0)
    }

    def runA_then_BC(A: SparseMatrix, B: SparseMatrix, C: SparseMatrix): (Long, Long) = {
      val t0  = nowMs
      val BC  = ExecutionEngine.spmm(B, C).map { case ((i, j), v) => (i, j, v) }
      val BCm = SparseMatrix(BC, B.nRows, C.nCols)
      val ABC = ExecutionEngine.spmm(A, BCm).map { case ((i, j), v) => (i, j, v) }
      val out = SparseMatrix(ABC, A.nRows, C.nCols)
      val nnz = out.entries.count()
      val t1  = nowMs
      (nnz, t1 - t0)
    }

    println("plan           | order       | outNNZ   | elapsed(ms) | est_first  | est_second | speedup_vs_alt | density")
    println("-----------------------------------------------------------------------------------------------------------")

    densities.foreach { dens =>
      val suffix = dstr(dens)
      val pathA = s"data/chain_matrix_${suffix}_A.csv"
      val pathB = s"data/chain_matrix_${suffix}_B.csv"
      val pathC = s"data/chain_matrix_${suffix}_C.csv"

      val A = Loader.loadCSVToCOO(sc, pathA)
      val B = Loader.loadCSVToCOO(sc, pathB)
      val C = Loader.loadCSVToCOO(sc, pathC)

      val plan = ChainPlanner.chooseOrder3(A, B, C)

      if (plan.order == "AB_then_C") runAB_then_C(A, B, C) else runA_then_BC(A, B, C)

      val (chosenNnz, chosenMs, chosenOrderLabel) =
        if (plan.order == "AB_then_C") {
          val (nnz, ms) = runAB_then_C(A, B, C)
          (nnz, ms, "AB_then_C")
        } else {
          val (nnz, ms) = runA_then_BC(A, B, C)
          (nnz, ms, "A_then_BC")
        }

      val (altNnz, altMs, altOrderLabel) =
        if (plan.order == "AB_then_C") {
          val (nnz, ms) = runA_then_BC(A, B, C)
          (nnz, ms, "A_then_BC")
        } else {
          val (nnz, ms) = runAB_then_C(A, B, C)
          (nnz, ms, "AB_then_C")
        }

      val speedupChosen = altMs.toDouble / chosenMs.toDouble
      val speedupAlt    = chosenMs.toDouble / altMs.toDouble

      println(f"${"chosen"}%14s | $chosenOrderLabel%-11s | $chosenNnz%8d | $chosenMs%10d | ${plan.firstCost}%.2e | ${plan.secondCost}%.2e | $speedupChosen%13.2f | ${dstr(dens)}")
      println(f"${"non-optimal"}%14s | $altOrderLabel%-11s | $altNnz%8d | $altMs%10d | ${plan.firstCost}%.2e | ${plan.secondCost}%.2e | $speedupAlt%13.2f | ${dstr(dens)}")

      val dims = s"${A.nRows},${A.nCols},${B.nRows},${B.nCols},${C.nRows},${C.nCols}"
      writer.println(s"${dstr(dens)},chosen,$chosenOrderLabel,$dims,$chosenNnz,$chosenMs,${plan.firstCost},${plan.secondCost},$speedupChosen")
      writer.println(s"${dstr(dens)},non-optimal,$altOrderLabel,$dims,$altNnz,$altMs,${plan.firstCost},${plan.secondCost},$speedupAlt")
    }

    writer.close()
    sc.stop()
    println(s"\n Chain-order benchmark results saved to ${outFile.getAbsolutePath}")
  }
}

package pdss

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import pdss.core._
import pdss.io.Loader
import pdss.engine.ExecutionEngine
import java.io.{File, PrintWriter}

object SpmmScaleBenchmark {

  private def nowMs: Long = System.nanoTime() / 1000000L

  def main(args: Array[String]): Unit = {

    // we test with 1,2,4,8 threads
    val threadLevels = Seq(1, 2, 4, 8)

    // CSV out
    val outFile = new File("spmm_scale_results.csv")
    val writer  = new PrintWriter(outFile)
    writer.println("threads,m,k,n,dens,partitions,outNNZ,elapsed_ms,speedup_vs_1,efficiency")

    // this MUST match what you already generated in data/
    val n        = 1000
    val dens     = 0.3
    // no f-string, no locale, just literal:
    val pathA    = "src/main/data/spmm_1000_0.300_A.csv"
    val pathB    = "src/main/data/spmm_1000_0.300_B.csv"
    val targetParts = 48

    println(s"🔎 will load: $pathA")
    println(s"🔎 will load: $pathB")
    println("threads | outNNZ | elapsed(ms) | speedup | eff")
    println("------------------------------------------------")

    var time1: Option[Long] = None

    for (threads <- threadLevels) {

      val conf = new SparkConf()
        .setAppName(s"SpmmScale-$threads")
        .setMaster(s"local[$threads]")
        .set("spark.serializer", "org.apache.spark.serializer.JavaSerializer")

      val spark = SparkSession.builder().config(conf).getOrCreate()
      val sc    = spark.sparkContext
      sc.setLogLevel("ERROR")

      // --- load matrices from disk (COO) ---
      val Araw = Loader.loadCOO(sc, pathA)
      val Braw = Loader.loadCOO(sc, pathB)

      // repartition + cache
      val A = SparseMatrix(Araw.entries.repartition(targetParts).cache(), Araw.nRows, Araw.nCols)
      val B = SparseMatrix(Braw.entries.repartition(targetParts).cache(), Braw.nRows, Braw.nCols)

      // materialize cache (not timed)
      A.entries.count()
      B.entries.count()

      // time SpMM (COO × COO)
      val t0 = nowMs
      val C  = ExecutionEngine.spmm(A, B)
      val outCount = C.count()
      val t1 = nowMs
      val elapsed = t1 - t0

      if (time1.isEmpty) time1 = Some(elapsed)
      val base       = time1.get.toDouble
      val speedup    = base / elapsed.toDouble
      val efficiency = speedup / threads.toDouble

      println(f"$threads%7d | $outCount%7d | $elapsed%10d | $speedup%7.2f | $efficiency%5.2f")

      writer.println(
        s"$threads,$n,$n,$n,$dens,$targetParts,$outCount,$elapsed,$speedup,$efficiency"
      )

      spark.stop()
    }

    writer.close()
    println(s"\n✅ SpMM scale results saved to ${outFile.getAbsolutePath}")
  }
}

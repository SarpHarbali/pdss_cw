package pdss

import org.apache.spark.sql.{SparkSession, functions => F}
import org.apache.spark.SparkConf
import pdss.core._
import pdss.io.Loader
import pdss.engine.ExecutionEngine
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, CoordinateMatrix, BlockMatrix}


import java.io.{File, PrintWriter}

object SpmmBenchmark {

  private def nowMs: Long = System.nanoTime() / 1000000L

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("SpmmBenchmark")
      .setMaster("local[*]")
      .set("spark.serializer", "org.apache.spark.serializer.JavaSerializer")

    val spark = SparkSession.builder().config(conf).getOrCreate()
    val sc    = spark.sparkContext
    sc.setLogLevel("ERROR")

    val sizes = Seq(250, 500, 750, 1000)
    val densities = Seq(0.1, 0.2, 0.3)

    val outFile = new File("spmm_benchmark_results.csv")
    val writer  = new PrintWriter(outFile)
    writer.println(
      "m,n,densityA,densityB,nnzA,nnzB," +
        "RDD_COO_ms,RDD_CSR_ms,DF_ms," +
        "COO_speedup_vs_DF,"
    )

    import spark.implicits._

    println("SpMM benchmark (from ./data)")
    println("m×n | dens | nnzA | nnzB | COO(ms) | CSR(ms) | DF(ms) | speedup(COO/DF)")
    println("---------------------------------------------------------------------------------------------------------------")


    for {
      n   <- sizes
      den <- densities
    } {
      val densStr = f"$den%1.3f"
      val pathA   = s"data/spmm_${n}_${densStr}_A.csv".replace(',', '.')
      val pathB   = s"data/spmm_${n}_${densStr}_B.csv".replace(',', '.')

      val fileA = new File(pathA)
      val fileB = new File(pathB)

      if (!fileA.exists() || !fileB.exists()) {
        println(s"Skipping $pathA / $pathB (file not found)")
      } else {
        val Acoo = Loader.loadCOO(sc, pathA)
        val Bcoo = Loader.loadCOO(sc, pathB)

        val nnzA = Acoo.entries.count()
        val nnzB = Bcoo.entries.count()

        val Acsr = Loader.cooToCSR(Acoo)
        val Bcsr = Loader.cooToCSR(Bcoo)

        val Bcsc = Loader.cooToCSC(Bcoo)


        val t1 = nowMs
        val cooRes   = ExecutionEngine.spmm(Acoo, Bcoo)
        val cooCount = cooRes.count()
        val t2 = nowMs
        val rddCooMs = t2 - t1


        val t3 = nowMs
        val csrRes   = ExecutionEngine.spmmCSRWithCSC(Acsr, Bcsc)
        val csrCount = csrRes.count()
        val t4 = nowMs
        val rddCsrMs = t4 - t3


        val dfA = Acoo.entries.toDF("i", "k", "vA")
        val dfB = Bcoo.entries.toDF("k", "j", "vB")

        dfA.createOrReplaceTempView("A")
        dfB.createOrReplaceTempView("B")

        val t7 = nowMs
        val dfCount = spark.sql(
          """
            |SELECT A.i AS i, B.j AS j, SUM(A.vA * B.vB) AS value
            |FROM A
            |JOIN B ON A.k = B.k
            |GROUP BY A.i, B.j
            |""".stripMargin
        ).count()
        val t8 = nowMs
        val dfMs = t8 - t7

        val csrSpeedup =
          if (rddCsrMs > 0) dfMs.toDouble / rddCsrMs.toDouble else Double.NaN

        println(
          f"${n}x$n | $den%1.3f | $nnzA%6d | $nnzB%6d | $rddCooMs%7d | $rddCsrMs%7d | $dfMs%7d | $csrSpeedup%6.2f"
        )

        writer.println(
          s"$n,$n,$den,$den,$nnzA,$nnzB," +
            s"$rddCooMs,$rddCsrMs,$dfMs," +
            s"$csrSpeedup"
        )
      }
    }

    writer.close()
    println(s"\n Results saved to ${outFile.getAbsolutePath}")

    spark.stop()
  }
}

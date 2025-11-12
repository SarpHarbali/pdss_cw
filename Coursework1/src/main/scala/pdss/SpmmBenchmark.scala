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

    // Spark
    val conf = new SparkConf()
      .setAppName("SpmmBenchmark")
      .setMaster("local[*]")
      .set("spark.serializer", "org.apache.spark.serializer.JavaSerializer")

    val spark = SparkSession.builder().config(conf).getOrCreate()
    val sc    = spark.sparkContext
    sc.setLogLevel("ERROR")

    // must match what DatasetGen produced
    val sizes = Seq(250, 500, 750, 1000)
    val densities = Seq(0.1, 0.2, 0.3)

    // CSV writer
    val outFile = new File("spmm_benchmark_results.csv")
    val writer  = new PrintWriter(outFile)
    writer.println(
      "m,k,n,densityA,densityB,nnzA,nnzB," +
        "RDD_COO_ms,RDD_CSR_ms,DF_ms,SQL_ms,MLLIB_ms," +                       // + MLLIB_ms
        "outNNZ_COO,outNNZ_CSR,outNNZ_DF,outNNZ_SQL,outNNZ_MLLIB," +           // + outNNZ_MLLIB
        "CSR_speedup_vs_DF,CSR_speedup_vs_MLLIB,DF_speedup_vs_MLLIB"            // + two new speedups
    )

    import spark.implicits._

    println("SpMM benchmark (from ./data)")
    println("m×n | dens | nnzA | nnzB | COO(ms) | CSR(ms) | DF(ms) | SQL(ms) | MLLIB(ms) | speedup(CSR/DF) | speedup(CSR/MLlib) | speedup(DF/MLlib)")
    println("---------------------------------------------------------------------------------------------------------------")


    for {
      n   <- sizes
      den <- densities
    } {
      // force dot decimal in filename
      val densStr = f"$den%1.3f".replace(',', '.') // e.g. "0.005"
      val pathA   = s"src/main/data/spmm_${n}_${densStr}_A.csv"
      val pathB   = s"src/main/data/spmm_${n}_${densStr}_B.csv"

      val fileA = new File(pathA)
      val fileB = new File(pathB)

      if (!fileA.exists() || !fileB.exists()) {
        println(s"⚠️  Skipping $pathA / $pathB (file not found)")
      } else {
        // 1) load COO
        val Acoo = Loader.loadCOO(sc, pathA)
        val Bcoo = Loader.loadCOO(sc, pathB)

        val nnzA = Acoo.entries.count()
        val nnzB = Bcoo.entries.count()

        // 2) convert to CSR (you said you already have CSR*CSR in engine)
        val Acsr = Loader.cooToCSR(Acoo)
        val Bcsr = Loader.cooToCSR(Bcoo)

        // ----------------------------------------------------
        // RDD: COO × COO
        // ----------------------------------------------------
        val t1 = nowMs
        val cooRes   = ExecutionEngine.spmm(Acoo, Bcoo)
        val cooCount = cooRes.count()
        val t2 = nowMs
        val rddCooMs = t2 - t1

        // ----------------------------------------------------
        // RDD: CSR × CSR   (calls your existing engine method)
        // ----------------------------------------------------
        val t3 = nowMs
        val csrRes   = ExecutionEngine.spmmCSR(Acsr, Bcsr)
        val csrCount = csrRes.count()
        val t4 = nowMs
        val rddCsrMs = t4 - t3

        // ----------------------------------------------------
        // DataFrame
        // ----------------------------------------------------
        val dfA = Acoo.entries.toDF("i", "k", "vA")
        val dfB = Bcoo.entries.toDF("k", "j", "vB")

        val t5 = nowMs
        val dfOut = dfA.join(dfB, "k")
          .select($"i", $"j", ($"vA" * $"vB").as("prod"))
          .groupBy($"i", $"j")
          .agg(F.sum($"prod").as("value"))
        val dfCount = dfOut.count()
        val t6 = nowMs
        val dfMs = t6 - t5

        // ----------------------------------------------------
        // SQL
        // ----------------------------------------------------
        dfA.createOrReplaceTempView("A")
        dfB.createOrReplaceTempView("B")

        val t7 = nowMs
        val sqlCount = spark.sql(
          """
            |SELECT A.i AS i, B.j AS j, SUM(A.vA * B.vB) AS value
            |FROM A
            |JOIN B ON A.k = B.k
            |GROUP BY A.i, B.j
            |""".stripMargin
        ).count()
        val t8 = nowMs
        val sqlMs = t8 - t7

        // ----------------------------------------------------
        // MLlib: BlockMatrix × BlockMatrix
        // ----------------------------------------------------
        val blockSize = 256

        // Build CoordinateMatrix from your COO RDDs
        val coordA = new CoordinateMatrix(
          Acoo.entries.map { case (i, k, v) => MatrixEntry(i.toLong, k.toLong, v) }
        )
        val coordB = new CoordinateMatrix(
          Bcoo.entries.map { case (k, j, v) => MatrixEntry(k.toLong, j.toLong, v) }
        )

        // Convert to BlockMatrix
        val bmA = coordA.toBlockMatrix(blockSize, blockSize)
        val bmB = coordB.toBlockMatrix(blockSize, blockSize)

        // Time multiply → force with count (like your other paths)
        val t9 = nowMs
        val bmC = bmA.multiply(bmB)
        val mllibCount = bmC.toCoordinateMatrix().entries.count()
        val t10 = nowMs
        val mllibMs = t10 - t9

        val csrVsMLlib     = if (mllibMs > 0) mllibMs.toDouble / rddCsrMs.toDouble else Double.NaN
        val dfVsMLlib      = if (mllibMs > 0) mllibMs.toDouble / dfMs.toDouble     else Double.NaN


        val csrSpeedup =
          if (rddCsrMs > 0) dfMs.toDouble / rddCsrMs.toDouble else Double.NaN

        println(
          f"${n}x$n | $den%1.3f | $nnzA%6d | $nnzB%6d | $rddCooMs%7d | $rddCsrMs%7d | $dfMs%7d | $sqlMs%7d | $mllibMs%9d | $csrSpeedup%6.2f | $csrVsMLlib%6.2f | $dfVsMLlib%6.2f"
        )

        writer.println(
          s"$n,$n,$n,$den,$den,$nnzA,$nnzB," +
            s"$rddCooMs,$rddCsrMs,$dfMs,$sqlMs,$mllibMs," +                    // + mllibMs
            s"$cooCount,$csrCount,$dfCount,$sqlCount,$mllibCount," +           // + mllibCount
            s"$csrSpeedup,$csrVsMLlib,$dfVsMLlib"
        )
      }
    }

    writer.close()
    println(s"\n✅ Results saved to ${outFile.getAbsolutePath}")

    spark.stop()
  }
}

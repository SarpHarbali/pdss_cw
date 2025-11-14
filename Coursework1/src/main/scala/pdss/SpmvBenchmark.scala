package pdss

import org.apache.spark.sql.{SparkSession, functions => F}
import org.apache.spark.SparkConf
import pdss.core._
import pdss.io.Loader
import java.io.{File, PrintWriter}

object SpmvBenchmarkFromData {

  private def nowMs: Long = System.nanoTime() / 1000000L

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("SpmvBenchmarkFromData")
      .setMaster("local[*]")
      .set("spark.serializer", "org.apache.spark.serializer.JavaSerializer")

    val spark = SparkSession.builder().config(conf).getOrCreate()
    val sc    = spark.sparkContext
    sc.setLogLevel("ERROR")

    val scenarios = Seq(
      (200, 0.005),
      (200, 0.010),
      (500, 0.005),
      (500, 0.010),
      (1000, 0.005),
      (1000, 0.010),
      (5000, 0.005),
      (5000, 0.010),
      (10000, 0.005),
      (10000, 0.010)
    )

    val outFile = new File("spmv_benchmark_results.csv")
    val writer  = new PrintWriter(outFile)
    writer.println("nRows,nCols,density,inputNNZ,RDD_COO_ms,RDD_CSR_ms,DataFrame_ms,SQL_ms,CSR_speedup_vs_DF")

    import spark.implicits._

    println("SpMV benchmark (loading from ./data)")
    println("Matrix     | dens  | nnz       | RDD-COO | RDD-CSR | DF | SQL | speedup")
    println("---------------------------------------------------------------------")

    for ((n, dens) <- scenarios) {
      val densStr = f"$dens%1.3f"
      val matPath = s"data/spmv_${n}_${densStr}.csv"
      val vecPath = s"data/spmv_vec_${n}.csv"

      val coo = Loader.loadCOO(sc, matPath)
      val vec = Loader.loadVector(sc, vecPath)
      val csr = Loader.cooToCSR(coo)

      val nnz = coo.entries.count().toInt

      val t1 = nowMs
      val y1 = pdss.engine.ExecutionEngine.spmv(coo, vec)
      y1.count()
      val t2 = nowMs
      val rddCooMs = t2 - t1

      val t3 = nowMs
      val y2 = pdss.engine.ExecutionEngine.spmvCSR(csr, vec)
      y2.count()
      val t4 = nowMs
      val rddCsrMs = t4 - t3

      val dfA = coo.entries.toDF("i", "j", "vA")
      val dfX = vec.values.toDF("j", "xj")

      val t5 = nowMs
      dfA.join(dfX, "j")
        .select($"i", ($"vA" * $"xj").as("prod"))
        .groupBy($"i")
        .agg(F.sum($"prod").as("value"))
        .count()
      val t6 = nowMs
      val dfMs = t6 - t5

      dfA.createOrReplaceTempView("A")
      dfX.createOrReplaceTempView("X")

      val t7 = nowMs
      spark.sql(
        """
          |SELECT A.i AS i, SUM(A.vA * X.xj) AS value
          |FROM A JOIN X ON A.j = X.j
          |GROUP BY A.i
          |""".stripMargin
      ).count()
      val t8 = nowMs
      val sqlMs = t8 - t7

      val csrSpeedup =
        if (rddCsrMs > 0) dfMs.toDouble / rddCsrMs.toDouble else Double.NaN

      println(f"${n}x$n%-8s | $dens%1.3f | $nnz%9d | $rddCooMs%7d | $rddCsrMs%7d | $dfMs%5d | $sqlMs%5d | $csrSpeedup%6.2f")

      writer.println(s"$n,$n,$dens,$nnz,$rddCooMs,$rddCsrMs,$dfMs,$sqlMs,$csrSpeedup")
    }

    writer.close()
    println(s"\n SpMV results saved to ${outFile.getAbsolutePath}")

    spark.stop()
  }
}

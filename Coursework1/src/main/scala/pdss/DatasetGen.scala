package pdss

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import java.io.{File, PrintWriter}
import scala.util.Random

object DatasetGen {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("DatasetGen").setMaster("local[*]")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    val outDir = "data"
    new File(outDir).mkdirs()

    val sizes = Seq(200, 500, 1000, 5000, 10000)
    val densities = Seq(0.005, 0.010)

    def writeCOO(path: String, entries: Seq[(Int, Int, Double)]): Unit = {
      val writer = new PrintWriter(new File(path))
      entries.foreach { case (i, j, v) => writer.println(s"$i,$j,$v") }
      writer.close()
    }

    def writeVec(path: String, values: Seq[(Int, Double)]): Unit = {
      val writer = new PrintWriter(new File(path))
      values.foreach { case (i, v) => writer.println(s"$i,$v") }
      writer.close()
    }

    for (n <- sizes; dens <- densities) {
      val nnz = (n.toLong * n.toLong * dens).toInt.max(1)
      val rnd = new Random(42 + n + (dens * 1000).toInt)
      val entries = (0 until nnz).map { _ =>
        val i = rnd.nextInt(n)
        val j = rnd.nextInt(n)
        val v = rnd.nextDouble() * 10
        (i, j, v)
      }

      val path = f"$outDir/spmv_${n}_${dens}%.3f.csv"
      writeCOO(path, entries)
      println(s"✅ Wrote matrix $path with $nnz entries")
    }

    for (n <- sizes) {
      val rnd = new Random(123 + n)
      val values = (0 until n).map(j => (j, rnd.nextDouble() * 5))
      val path = s"$outDir/spmv_vec_${n}.csv"
      writeVec(path, values)
      println(s"✅ Wrote vector $path with ${values.size} entries")
    }

    spark.stop()
    println(s"\n✅ All datasets written to $outDir/")
  }
}

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

    val outDir = "src/main/data"
    new File(outDir).mkdirs()

    val sizes = Seq(250, 500, 750, 1000, 1500, 3000)
    val densities = Seq(0.1, 0.2, 0.3)

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

      // Matrix A
      val rndA = new Random(42 + n + (dens * 1000).toInt)
      val entriesA = (0 until nnz).map { _ =>
        val i = rndA.nextInt(n)
        val j = rndA.nextInt(n)
        val v = rndA.nextDouble() * 10
        (i, j, v)
      }
      val pathA = f"$outDir/spmm_${n}_${dens}%.3f_A.csv".replace(',', '.')

      writeCOO(pathA, entriesA)
      println(s"✅ Wrote matrix $pathA with $nnz entries")

      // Matrix B
      val rndB = new Random(84 + n + (dens * 1000).toInt)
      val entriesB = (0 until nnz).map { _ =>
        val i = rndB.nextInt(n)
        val j = rndB.nextInt(n)
        val v = rndB.nextDouble() * 10
        (i, j, v)
      }
      val pathB = f"$outDir/spmm_${n}_${dens}%.3f_B.csv".replace(',', '.')
      writeCOO(pathB, entriesB)
      println(s"✅ Wrote matrix $pathB with $nnz entries")
    }

//    for (n <- sizes) {
//      val rnd = new Random(123 + n)
//      val values = (0 until n).map(j => (j, rnd.nextDouble() * 5))
//      val path = s"$outDir/spmv_vec_${n}.csv"
//      writeVec(path, values)
//      println(s"✅ Wrote vector $path with ${values.size} entries")
//    }

    spark.stop()
    println(s"\n✅ All datasets written to $outDir/")
  }
}

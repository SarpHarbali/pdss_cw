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

    val sizes = Seq(5000, 10000)
    val densities = Seq(0.001, 0.002)

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

      val rndA = new Random(42 + n + (dens * 1000).toInt)
      val entriesA = (0 until nnz).map { _ =>
        val i = rndA.nextInt(n)
        val j = rndA.nextInt(n)
        val v = rndA.nextDouble() * 10
        (i, j, v)
      }
      val pathA = f"$outDir/spmm_${n}_${dens}%.3f_A.csv"

      writeCOO(pathA, entriesA)
      println(s" Wrote matrix $pathA with $nnz entries")

      val rndB = new Random(84 + n + (dens * 1000).toInt)
      val entriesB = (0 until nnz).map { _ =>
        val i = rndB.nextInt(n)
        val j = rndB.nextInt(n)
        val v = rndB.nextDouble() * 10
        (i, j, v)
      }
      val pathB = f"$outDir/spmm_${n}_${dens}%.3f_B.csv"
      writeCOO(pathB, entriesB)
      println(s" Wrote matrix $pathB with $nnz entries")
    }

    val chainDensities = Seq(0.1, 0.2, 0.3)

    def densStr(d: Double): String = f"$d%.3f"

    case class MatSpec(label: String, rows: Int, cols: Int, seedBase: Int)

    val chainSpecs = Seq(
      MatSpec("A", 50000, 10,   12021),
      MatSpec("B",   10, 50000, 22021),
      MatSpec("C", 50000, 10,   32021)
    )

    for (dens <- chainDensities; spec <- chainSpecs) {
      val rows = spec.rows
      val cols = spec.cols
      val nnz = math.max(1, (rows.toLong * cols.toLong * dens).toInt)

      val rnd = new Random(spec.seedBase + (dens * 1000).toInt)

      val entries = (0 until nnz).map { _ =>
        val i = rnd.nextInt(rows)
        val j = rnd.nextInt(cols)
        val v = rnd.nextDouble() * 10
        (i, j, v)
      }

      val path = s"$outDir/spmm_chain_${spec.label}_${rows}x${cols}_${densStr(dens)}.csv"
      writeCOO(path, entries)

    }
    spark.stop()
    println(s"\n All datasets written to $outDir/")
  }
}

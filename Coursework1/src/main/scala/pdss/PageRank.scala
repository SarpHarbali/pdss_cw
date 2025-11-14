package pdss

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import pdss.core._
import pdss.io.Loader
import pdss.engine.ExecutionEngine

object PageRank extends App {

  val alpha   = 0.85
  val iters   = 10

  val conf = new SparkConf()
    .setAppName("PDSS-PageRank")
    .setMaster(s"local[*]")
    .set("spark.ui.enabled","false")

  val sc = new SparkContext(conf)
  sc.setLogLevel("WARN")



  import java.io.File
  import scala.io.Source

  val pathAdj = "src/main/data/pagerankfile.csv"
  val abs = new File(pathAdj).getAbsolutePath
  val A = Loader.loadCOO(sc, abs)
  val n: Long = A.nRows
  val nInt: Int = n.toInt

  val nodesRDD = A.entries.flatMap { case (i, j, _) => Seq(i, j) }.distinct().cache()
  val minId = nodesRDD.min()
  val maxId = nodesRDD.max()
  val nDistinct = nodesRDD.count()
  val edgesCount = A.entries.count()

  println(s"loaded from: $pathAdj")
  println(s"nodes: distinct=$nDistinct, min=$minId, max=$maxId, edges=$edgesCount")
  require(minId == 0, s"Expected minId=0 but got $minId")
  require(maxId == 99, s"Expected maxId=99 but got $maxId ")

  val outDeg: RDD[(Int, Long)] =
    A.entries
      .map{ case (i, _, _) => (i, 1L) }
      .reduceByKey(_ + _)
      .persist()

  val M_entries: RDD[(Int, Int, Double)] =
    A.entries
      .keyBy{ case (i, j, v) => i }
      .join(outDeg)
      .map{ case (_, ((i, j, _), d)) => (i, j, 1.0 / d.toDouble) }

  val M = SparseMatrix(M_entries, nRows = A.nRows, nCols = A.nCols)

  val MT = {
    val mtEntries = M.entries.map { case (i, j, v) => (j, i, v) }
    SparseMatrix(mtEntries, nRows = M.nCols, nCols = M.nRows)
  }


  val allNodes: RDD[(Int, Double)] =
    sc.parallelize(0 until nInt).map(i => (i, 0.0)).persist()


  val nodesWithOut: RDD[(Int, Int)] = outDeg.map { case (i, _) => (i, 1) }
  val danglingFlags: RDD[(Int, Int)] =
    allNodes
      .keys
      .map(i => (i, 0))
      .leftOuterJoin(nodesWithOut)
      .mapValues { case (_, hasOutOpt) => if (hasOutOpt.isDefined) 0 else 1 }
      .persist()


  val r0 = {
    val vals = sc.parallelize(0 until nInt).map(i => (i, 1.0 / n.toDouble))
    DistVector(vals, length = n)
  }


  var r = r0
  for (_ <- 1 to iters) {


    val y: RDD[(Int, Double)] = ExecutionEngine.spmv(MT, r)


    val danglingMass: Double =
      r.values
        .join(danglingFlags)
        .filter { case (_, (_, flag)) => flag == 1 }
        .map { case (_, (ri, _)) => ri }
        .sum()

    val teleBase = (1.0 - alpha) / n.toDouble
    val teleDang = alpha * danglingMass / n.toDouble


    val yAll: RDD[(Int, Double)] =
      allNodes
        .leftOuterJoin(y)
        .mapValues { case (_, opt) => opt.getOrElse(0.0) }


    val rNewVals: RDD[(Int, Double)] =
      yAll.mapValues(v => alpha * v + teleBase + teleDang)

    r = DistVector(rNewVals, length = n)
  }

  val sumRank = r.values.map(_._2).sum()
  println(f"Iterations: $iters")
  println(f"Sum of ranks ≈ $sumRank%.6f")

  val topK = r.values
    .map{ case (i, v) => (v, i) }
    .top(10)
    .map{ case (v, i) => (i, v) }

  println("Top 10 nodes by PageRank:")
  topK.foreach{ case (i, v) => println(f"  node=$i%5d   rank=$v%.6e") }

  sc.stop()
}

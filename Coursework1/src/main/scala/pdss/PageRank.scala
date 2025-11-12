package pdss

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import pdss.core._
import pdss.io.Loader
import pdss.engine.ExecutionEngine

object PageRank extends App {

  // -------------------------------
  // Config
  // -------------------------------
  val alpha   = 0.85                              // damping
  val iters   = 10                                // iterations
  val threads = "*"                               // local concurrency

  val conf = new SparkConf()
    .setAppName("PDSS-PageRank")
    .setMaster(s"local[*]")
    .set("spark.ui.enabled","false")

  val sc = new SparkContext(conf)
  sc.setLogLevel("WARN")

  // -------------------------------
  // Load adjacency as COO (i,j,v)
  // -------------------------------

  import java.io.File
  import scala.io.Source

  val pathAdj = "src/main/data/pagerankfile.csv"   // whatever you're using
  val abs = new File(pathAdj).getAbsolutePath
  val A = Loader.loadCOO(sc, abs)
  val n: Long = A.nRows                     // number of nodes (rows)
  val nInt: Int = n.toInt

  val nodesRDD = A.entries.flatMap { case (i, j, _) => Seq(i, j) }.distinct().cache()
  val minId = nodesRDD.min()
  val maxId = nodesRDD.max()
  val nDistinct = nodesRDD.count()
  val edgesCount = A.entries.count()

  println(s"[CHECK] loaded from: $pathAdj")
  println(s"[CHECK] nodes: distinct=$nDistinct, min=$minId, max=$maxId, edges=$edgesCount")
  require(minId == 0, s"Expected minId=0 but got $minId")
  require(maxId == 99, s"Expected maxId=99 but got $maxId ")

  // out-degree per node (for normalization + dangling detection)
  val outDeg: RDD[(Int, Long)] =
    A.entries
      .map{ case (i, _, _) => (i, 1L) }
      .reduceByKey(_ + _)
      .persist()

  // Build row-stochastic M = D^-1 * A : each outgoing edge gets weight 1/deg(u)
  val M_entries: RDD[(Int, Int, Double)] =
    A.entries
      .keyBy{ case (i, j, v) => i }                 // (i, (i,j,v))
      .join(outDeg)                                 // (i, ((i,j,v), deg))
      .map{ case (_, ((i, j, _), d)) => (i, j, 1.0 / d.toDouble) }

  val M = SparseMatrix(M_entries, nRows = A.nRows, nCols = A.nCols)

  // Pre-build M^T once (avoid per-iteration swaps)
  val MT = {
    val mtEntries = M.entries.map { case (i, j, v) => (j, i, v) }
    SparseMatrix(mtEntries, nRows = M.nCols, nCols = M.nRows)
  }

  // -------------------------------
  // Helpers for full-vector ops
  // -------------------------------
  // "All nodes" skeleton so we can materialize dense vectors when needed
  val allNodes: RDD[(Int, Double)] =
    sc.parallelize(0 until nInt).map(i => (i, 0.0)).persist()

  // Dangling nodes = nodes with out-degree 0
  // Create flags: (i, 1) if dangling, else (i, 0)
  val nodesWithOut: RDD[(Int, Int)] = outDeg.map { case (i, _) => (i, 1) }
  val danglingFlags: RDD[(Int, Int)] =
    allNodes
      .keys
      .map(i => (i, 0))                    // base 0
      .leftOuterJoin(nodesWithOut)         // (i, (0, Some(1)/None))
      .mapValues { case (_, hasOutOpt) => if (hasOutOpt.isDefined) 0 else 1 }
      .persist()

  // -------------------------------
  // Initial rank vector r0 = 1/N
  // -------------------------------
  val r0 = {
    val vals = sc.parallelize(0 until nInt).map(i => (i, 1.0 / n.toDouble))
    DistVector(vals, length = n)
  }

  // -------------------------------
  // Power iterations:
  // r_{t+1} = alpha * (M^T r_t) + (1 - alpha)/N + alpha * (dangling_mass)/N
  // -------------------------------
  var r = r0
  for (_ <- 1 to iters) {

    // Multiply: y = M^T * r
    // (ExecutionEngine.spmv expects SparseMatrix x DistVector -> RDD[(Int,Double)])
    val y: RDD[(Int, Double)] = ExecutionEngine.spmv(MT, r)

    // Sum of rank mass at dangling nodes (out-degree == 0)
    val danglingMass: Double =
      r.values
        .join(danglingFlags)              // (i, (ri, flag))
        .filter { case (_, (_, flag)) => flag == 1 }
        .map { case (_, (ri, _)) => ri }
        .sum()

    val teleBase = (1.0 - alpha) / n.toDouble
    val teleDang = alpha * danglingMass / n.toDouble

    // Ensure every node is present in y (fill missing with 0.0)
    val yAll: RDD[(Int, Double)] =
      allNodes
        .leftOuterJoin(y)                 // (i, (0.0, Some/None))
        .mapValues { case (_, opt) => opt.getOrElse(0.0) }

    // Combine: rNew = alpha * yAll + teleBase + teleDang  (elementwise)
    val rNewVals: RDD[(Int, Double)] =
      yAll.mapValues(v => alpha * v + teleBase + teleDang)

    r = DistVector(rNewVals, length = n)
  }

  // -------------------------------
  // Diagnostics
  // -------------------------------
  val sumRank = r.values.map(_._2).sum()
  println(f"Iterations: $iters")
  println(f"Sum of ranks ≈ $sumRank%.6f")

  // Show top-10 nodes by rank (optional)
  val topK = r.values
    .map{ case (i, v) => (v, i) }
    .top(10)   // by value
    .map{ case (v, i) => (i, v) }

  println("Top 10 nodes by PageRank:")
  topK.foreach{ case (i, v) => println(f"  node=$i%5d   rank=$v%.6e") }

  sc.stop()
}

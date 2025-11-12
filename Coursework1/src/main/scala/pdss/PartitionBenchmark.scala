package pdss

import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.rdd.RDD
import pdss.core._
import pdss.io.Loader

object PartitionBenchmark extends App {

  val pathA = "src/main/data/spmm_3000_0.200_A.csv"
  val pathB = "src/main/data/spmm_3000_0.200_B.csv"
  val threads = 8
  val trials = 1
  val partsToTest = List(8, 16, 32, 64, 128)

  def median(xs: Seq[Double]) = xs.sorted.apply(xs.size / 2)

  def spmmWithParts(A: SparseMatrix, B: SparseMatrix, parts: Int): RDD[((Int, Int), Double)] = {
    val AbyK = A.entries.map { case (i, k, vA) => (k, (i, vA)) }
    val BbyK = B.entries.map { case (k, j, vB) => (k, (j, vB)) }
    val p = new HashPartitioner(parts)

    println(s"   → Co-partitioning inputs with $parts partitions ...")
    val L = AbyK.partitionBy(p).persist()
    val R = BbyK.partitionBy(p).persist()
    val joined = L.join(R)
    val products = joined.map { case (_, ((i, vA), (j, vB))) => ((i, j), vA * vB) }
    val out = products.reduceByKey(p, _ + _)

    L.unpersist(false); R.unpersist(false)
    out
  }

  case class Row(parts: Int, seconds: Double)

  val conf = new SparkConf()
    .setAppName("PDSS-Partitions")
    .setMaster(s"local[$threads]")
    .set("spark.ui.enabled", "false")
  val sc = new SparkContext(conf)
  sc.setLogLevel("WARN")

  println(s"🔹 Loading matrices with $threads Spark threads ...")
  val A = Loader.loadCSVToCOO(sc, pathA)
  val B = Loader.loadCSVToCOO(sc, pathB)
  A.entries.persist(StorageLevel.MEMORY_ONLY)
  B.entries.persist(StorageLevel.MEMORY_ONLY)
  A.entries.count(); B.entries.count()

  println("→ Warm-up run ...")
  spmmWithParts(A, B, parts = threads).count()

  val rows = partsToTest.map { parts =>
    println(s"\n==============================")
    println(s"🔸 Running test for $parts partitions")
    println(s"==============================")
    val times = (1 to trials).map { run =>
      println(s"   ⏱️ Run #$run ...")
      val t0 = System.nanoTime()
      val nnz = spmmWithParts(A, B, parts).count()
      val t1 = System.nanoTime()
      val sec = (t1 - t0) / 1e9
      println(f"   ✅ Finished run #$run in $sec%.3f sec (nnz=$nnz)")
      sec
    }
    val medianTime = median(times)
    println(f"✅ Done with $parts partitions | median = $medianTime%.3f sec")
    Row(parts, medianTime)
  }

  sc.stop()

  println("\n=== Final Results ===")
  println("partitions,seconds")
  rows.foreach(r => println(f"${r.parts},${r.seconds}%.6f"))
}
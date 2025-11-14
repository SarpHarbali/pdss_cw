package pdss

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.storage.StorageLevel
import pdss.core.{DenseMatrix, SparseTensor}
import pdss.engine.TensorEngine

import scala.collection.mutable
import scala.util.Random

object TensorMttkrpBenchmark {

  private case class Scenario(
      name: String,
      shape: Array[Int],
      density: Double,
      rank: Int,
      targetMode: Int,
      trials: Int
  )

  private case class Overrides(
      shape: Option[Array[Int]] = None,
      density: Option[Double] = None,
      rank: Option[Int] = None,
      targetMode: Option[Int] = None,
      trials: Option[Int] = None
  )

  private val defaultScenarios = Seq(
    Scenario("cube_300", Array(300, 300, 300), density = 0.0002, rank = 32, targetMode = 0, trials = 3),
    Scenario("tall_skinny", Array(200, 400, 800), density = 0.0001, rank = 32, targetMode = 1, trials = 3)
  )

  def main(args: Array[String]): Unit = {
    val overrides = parseOverrides(args.toSeq)

    val scenarios = overrides.shape match {
      case Some(shape) =>
        Seq(
          Scenario(
            name = overridesLabel(shape, overrides),
            shape = shape,
            density = overrides.density.getOrElse(0.0002),
            rank = overrides.rank.getOrElse(32),
            targetMode = overrides.targetMode.getOrElse(0),
            trials = overrides.trials.getOrElse(3)
          )
        )
      case None =>
        defaultScenarios.map { scn =>
          scn.copy(
            density = overrides.density.getOrElse(scn.density),
            rank = overrides.rank.getOrElse(scn.rank),
            targetMode = overrides.targetMode.getOrElse(scn.targetMode),
            trials = overrides.trials.getOrElse(scn.trials)
          )
        }
    }

    val conf = new SparkConf()
      .setAppName("Tensor-MTTKRP-Benchmark")
      .setMaster("local[*]")
      .set("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val out = new java.io.PrintWriter("tensor_mttkrp_results.csv")
    out.println("scenario,shape,density,rank,targetMode,algo,trial,time_ms,out_rows,nnz")

    println("Tensor MTTKRP benchmark (COO vs CSF)")
    println("scenario | algo | trial | time(ms) | nnz | outRows | speedup vs prev")
    println("---------------------------------------------------------------------")

    try {
      scenarios.foreach { scenario =>
        runScenario(sc, scenario, out)
      }
    } finally {
      out.close()
      sc.stop()
    }
  }

  private def runScenario(
      sc: SparkContext,
      scenario: Scenario,
      out: java.io.PrintWriter
  ): Unit = {
    val partitions = math.max(sc.defaultParallelism, 8)

    val (tensor, nnz) = randomTensor(sc, scenario.shape, scenario.density, seed = scenario.hashCode.toLong, partitions)
    val factors = scenario.shape.indices.map { mode =>
      randomDenseMatrix(sc, scenario.shape(mode), scenario.rank, seed = (mode + 1) * 1771L + scenario.hashCode, partitions)
    }

    val target = scenario.targetMode

    val factorRdds = factors.map(_.rows.persist(StorageLevel.MEMORY_ONLY))
    factorRdds.foreach(_.count())

    val cooTimes = measureTrials(scenario.trials) {
      TensorEngine.mttkrpCoo(tensor, factors, target)
    }
    val csfTimes = measureTrials(scenario.trials) {
      TensorEngine.mttkrp(tensor, factors, target)
    }

    cooTimes.zipWithIndex.foreach { case ((timeMs, outRows), idx) =>
      val trial = idx + 1
      out.println(s"${scenario.name},${scenario.shape.mkString("x")},${scenario.density},${scenario.rank},${target},COO,$trial,$timeMs,$outRows,$nnz")
      println(f"${scenario.name}%-10s | COO | $trial%2d | $timeMs%8d | $nnz%8d | $outRows%7d |    -")
    }

    csfTimes.zipWithIndex.foreach { case ((timeMs, outRows), idx) =>
      val trial = idx + 1
      val paired = if (idx < cooTimes.length) cooTimes(idx)._1.toDouble else Double.NaN
      val speedup = if (paired.isNaN || timeMs == 0L) Double.NaN else paired / timeMs.toDouble
      out.println(s"${scenario.name},${scenario.shape.mkString("x")},${scenario.density},${scenario.rank},${target},CSF,$trial,$timeMs,$outRows,$nnz")
      if (speedup.isNaN) {
        println(f"${scenario.name}%-10s | CSF | $trial%2d | $timeMs%8d | $nnz%8d | $outRows%7d |    -")
      } else {
        println(f"${scenario.name}%-10s | CSF | $trial%2d | $timeMs%8d | $nnz%8d | $outRows%7d | ${speedup}%6.2f×")
      }
    }

    tensor.entries.unpersist()
    factorRdds.foreach(_.unpersist())
  }

  private def measureTrials(trials: Int)(op: => org.apache.spark.rdd.RDD[(Int, Array[Double])]): Seq[(Long, Long)] = {
    (1 to trials).map { _ =>
      val start = System.nanoTime()
      val result = op
      val rows = result.count()
      val end = System.nanoTime()
      val elapsedMs = ((end - start) / 1e6).toLong
      (elapsedMs, rows)
    }
  }

  private def randomTensor(
      sc: SparkContext,
      shape: Array[Int],
      density: Double,
      seed: Long,
      partitions: Int
  ): (SparseTensor, Long) = {
    val totalCoords = shape.map(_.toLong).product
    val nnz = math.max(1L, math.round(totalCoords.toDouble * density))
    val ids = sc.range(0L, nnz, 1L, partitions)
    val entries = ids.mapPartitionsWithIndex { case (pid, iter) =>
      val rnd = new Random(seed + pid)
      iter.map { _ =>
        val coords = new Array[Int](shape.length)
        var d = 0
        while (d < shape.length) {
          coords(d) = rnd.nextInt(shape(d))
          d += 1
        }
        val value = rnd.nextDouble()
        (coords, value)
      }
    }.persist(StorageLevel.MEMORY_ONLY)
    val count = entries.count()
    (SparseTensor(entries, shape.clone()), count)
  }

  private def randomDenseMatrix(
      sc: SparkContext,
      rows: Int,
      cols: Int,
      seed: Long,
      partitions: Int
  ): DenseMatrix = {
    val data = sc.parallelize(0 until rows, math.max(1, partitions)).mapPartitionsWithIndex { case (pid, iter) =>
      val rnd = new Random(seed + pid)
      iter.map { rowIdx =>
        val values = Array.fill(cols)(rnd.nextDouble())
        (rowIdx, values)
      }
    }
    DenseMatrix(data, rows, cols)
  }

  private def parseOverrides(args: Seq[String]): Overrides = {
    if (args.isEmpty) return Overrides()
    val map = mutable.Map.empty[String, String]
    args.foreach { arg =>
      val parts = arg.split('=')
      if (parts.length == 2 && parts.head.startsWith("--")) {
        map += parts.head.drop(2) -> parts(1)
      }
    }
    Overrides(
      shape = map.get("shape").map(_.split(',').map(_.trim.toInt)),
      density = map.get("density").map(_.toDouble),
      rank = map.get("rank").map(_.toInt),
      targetMode = map.get("target").map(_.toInt),
      trials = map.get("trials").map(_.toInt)
    )
  }

  private def overridesLabel(shape: Array[Int], overrides: Overrides): String = {
    val shapeStr = shape.mkString("x")
    val extras = List(
      overrides.density.map(d => f"dens=$d%g"),
      overrides.rank.map(r => s"r=$r"),
      overrides.targetMode.map(t => s"mode=$t")
    ).flatten
    if (extras.isEmpty) shapeStr else s"${shapeStr}_${extras.mkString("_")}"
  }
}

package imdb

import org.apache.spark.rdd.RDD
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.BeforeAndAfterAll

class MainTest extends AnyFlatSpec with Matchers with BeforeAndAfterAll {
  private var titleBasicsRDD: RDD[TitleBasics] = _
  private var titleRatingsRDD: RDD[TitleRatings] = _
  private var titleCrewRDD: RDD[TitleCrew] = _
  private var nameBasicsRDD: RDD[NameBasics] = _

  override def beforeAll(): Unit = {
    super.beforeAll()

    // Configure logging
    org.apache.log4j.Logger.getLogger("org").setLevel(org.apache.log4j.Level.ERROR)
    org.apache.log4j.Logger.getLogger("akka").setLevel(org.apache.log4j.Level.ERROR)

    // Initialize RDDs
    titleBasicsRDD = Main.titleBasicsRDD
    titleRatingsRDD = Main.titleRatingsRDD
    titleCrewRDD = Main.titleCrewRDD
    nameBasicsRDD = Main.nameBasicsRDD
  }

  override def afterAll(): Unit = {
    SparkContextWrapper.stop()
    super.afterAll()
  }

  private def printTestStatus(taskName: String, executionTime: Long, passed: Boolean): Unit = {
    val statusSymbol = if (passed) "✓" else "✗"
    println(s"""
      |----------------------------------------
      |Test Results for $taskName:
      |Status: $statusSymbol ${if (passed) "PASSED" else "FAILED"}
      |Execution Time: ${executionTime}ms
      |----------------------------------------
      |""".stripMargin)
  }

  "Task1: Genre Growth Analysis" should "find the correct top 5 genres with growth" in {
    val expected = List(
      ("Drama", 3504),
      ("Documentary", 2883),
      ("Comedy", 2439),
      ("Horror", 909),
      ("Thriller", 854)
    )

    val startTime = System.currentTimeMillis()
    val result = Main.task1(titleBasicsRDD).collect().toList
    val endTime = System.currentTimeMillis()
    val executionTime = endTime - startTime

    // Verify results
    val correctLength = result.length == 5
    val correctContent = result == expected

    val passed = correctLength && correctContent
    printTestStatus("Task 1", executionTime, passed)

    if (!correctLength) println(s"Expected length: 5, but got: ${result.length}")
    if (!correctContent) {
      println("Content mismatch. Expected vs Actual:")
      expected.zip(result).foreach { case (exp, res) =>
        println(s"Expected: $exp, Got: $res")
      }
    }

    result should have length 5
    result should contain theSameElementsInOrderAs expected
  }

  "Task2: Genre Ratings by Decade" should "calculate correct weighted averages per decade" in {
    val expected = List(
      (1900, List(("Music", 6.597943f), ("Crime", 6.159037f), ("Biography", 6.0411067f))),
      (1910, List(("War", 7.1199822f), ("Horror", 6.914935f), ("Thriller", 6.7857585f))),
      (1920, List(("Sci-Fi", 8.284826f), ("Family", 8.154517f), ("Action", 8.095828f))),
      (1930, List(("Thriller", 8.187101f), ("Mystery", 7.9939985f), ("Crime", 7.950047f))),
      (1940, List(("Fantasy", 8.493501f), ("Family", 8.429712f), ("War", 8.406213f))),
      (1950, List(("Crime", 8.504564f), ("Thriller", 8.394195f), ("Mystery", 8.336316f))),
      (1960, List(("Western", 8.649012f), ("Adventure", 8.299024f), ("Thriller", 8.269708f))),
      (1970, List(("Crime", 8.794421f), ("Drama", 8.321298f), ("Sci-Fi", 8.06072f))),
      (1980, List(("War", 8.122867f), ("Western", 7.992603f), ("Sci-Fi", 7.978767f))),
      (1990, List(("History", 8.594199f), ("Biography", 8.496143f), ("Drama", 8.360406f))),
      (2000, List(("History", 8.6177435f), ("War", 8.295243f), ("Drama", 8.035681f))),
      (2010, List(("Western", 8.212091f), ("History", 7.9701552f), ("Animation", 7.8727164f))),
      (2020, List(("Animation", 7.9739513f), ("Adventure", 7.5102105f), ("Action", 7.332238f)))
    )

    val startTime = System.currentTimeMillis()
    val result = Main.task2(titleBasicsRDD, titleRatingsRDD).collect().toList
    val endTime = System.currentTimeMillis()
    val executionTime = endTime - startTime

    // Verify results
    var allMatches = true
    var mismatchDetails = new StringBuilder()

    result.zip(expected).foreach { case ((decadeR, genresR), (decadeE, genresE)) =>
      if (decadeR != decadeE) {
        allMatches = false
        mismatchDetails.append(s"\nDecade mismatch: Expected $decadeE, Got $decadeR")
      }
      if (genresR.length != 3) {
        allMatches = false
        mismatchDetails.append(s"\nIncorrect number of genres for decade $decadeR: Expected 3, Got ${genresR.length}")
      }
      genresR.zip(genresE).foreach { case ((genreR, ratingR), (genreE, ratingE)) =>
        if (genreR != genreE) {
          allMatches = false
          mismatchDetails.append(s"\nGenre mismatch in decade $decadeR: Expected $genreE, Got $genreR")
        }
        if (math.abs(ratingR - ratingE) > 0.0001f) {
          allMatches = false
          mismatchDetails.append(s"\nRating mismatch for $genreR in decade $decadeR: Expected $ratingE, Got $ratingR")
        }
      }
    }

    val passed = allMatches && result.length == expected.length
    printTestStatus("Task 2", executionTime, passed)

    if (!allMatches) println(s"Mismatches found: $mismatchDetails")
    if (result.length != expected.length) println(s"Length mismatch: Expected ${expected.length}, Got ${result.length}")

    // Original assertions
    result.length shouldBe expected.length
    result.zip(expected).foreach { case ((decadeR, genresR), (decadeE, genresE)) =>
      decadeR shouldBe decadeE
      genresR.length shouldBe 3
      genresR.zip(genresE).foreach { case ((genreR, ratingR), (genreE, ratingE)) =>
        genreR shouldBe genreE
        ratingR shouldBe (ratingE +- 0.0001f)
      }
    }
  }

  "Task3: Top Directors" should "find directors with highest rated films" in {
    val expected = List(
      ("Tetsurô Araki", 9.2f),
      ("Matt Shakman", 9.066667f)
    )

    val startTime = System.currentTimeMillis()
    val result = Main.task3(titleBasicsRDD, titleRatingsRDD, titleCrewRDD).collect().toList
    val endTime = System.currentTimeMillis()
    val executionTime = endTime - startTime

    // Verify results
    val correctLength = result.length == 2
    var correctContent = true
    var mismatchDetails = new StringBuilder()

    result.zip(expected).foreach { case ((nameR, ratingR), (nameE, ratingE)) =>
      if (nameR != nameE) {
        correctContent = false
        mismatchDetails.append(s"\nName mismatch: Expected $nameE, Got $nameR")
      }
      if (math.abs(ratingR - ratingE) > 0.0001f) {
        correctContent = false
        mismatchDetails.append(s"\nRating mismatch for $nameR: Expected $ratingE, Got $ratingR")
      }
    }

    val passed = correctLength && correctContent
    printTestStatus("Task 3", executionTime, passed)

    if (!correctLength) println(s"Length mismatch: Expected 2, Got ${result.length}")
    if (!correctContent) println(s"Mismatches found: $mismatchDetails")

    // Original assertions
    result should have length 2
    result.zip(expected).foreach { case ((nameR, ratingR), (nameE, ratingE)) =>
      nameR shouldBe nameE
      ratingR shouldBe (ratingE +- 0.0001f)
    }
  }

  "Task4: Prolific Crew Members" should "identify top contributors across time periods" in {
    val expected = List(
      ("Marco Romano", 12),
      ("Jordan Hill", 12),
      ("Sergey A.", 10),
      ("Tyler Perry", 10),
      ("Stan Lee", 10),
      ("Dustin Ferguson", 9),
      ("Matthew A. Peters", 8),
      ("Jack Kirby", 8),
      ("James Ian Mair", 8),
      ("Tony Newton", 8)
    )

    val startTime = System.currentTimeMillis()
    val result = Main.task4(titleBasicsRDD, nameBasicsRDD, titleCrewRDD).collect().toList
    val endTime = System.currentTimeMillis()
    val executionTime = endTime - startTime

    // Verify results
    val correctLength = result.length == 10
    val correctContent = result == expected

    val passed = correctLength && correctContent
    printTestStatus("Task 4", executionTime, passed)

    if (!correctLength) println(s"Length mismatch: Expected 10, Got ${result.length}")
    if (!correctContent) {
      println("Content mismatch. Expected vs Actual:")
      expected.zip(result).foreach { case (exp, res) =>
        println(s"Expected: $exp, Got: $res")
      }
    }

    // Original assertions
    result should have length 10
    result should contain theSameElementsInOrderAs expected
  }
}

// SparkContext wrapper remains the same
object SparkContextWrapper {
  import org.apache.spark.{SparkConf, SparkContext}

  private var instance: SparkContext = _

  def getInstance(): SparkContext = {
    if (instance == null || instance.isStopped) {
      val conf = new SparkConf()
        .setAppName("ImdbAnalysisTest")
        .setMaster("local[*]")
        .set("spark.ui.enabled", "false")
      instance = new SparkContext(conf)
    }
    instance
  }

  def stop(): Unit = {
    if (instance != null && !instance.isStopped) {
      instance.stop()
      instance = null
    }
  }
}

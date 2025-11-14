package pdss

import java.io.File
import java.nio.file.Files

import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import pdss.core.{DenseMatrix, SparseTensor}
import pdss.engine.TensorEngine
import pdss.frontend.LinearAlgebraAPI

class TensorEngineTest extends AnyFlatSpec with Matchers with BeforeAndAfterAll {

	private var sc: SparkContext = _
	private var tempHadoopDir: File = _

	override protected def beforeAll(): Unit = {
		super.beforeAll()
		ensureWinutilsStub()
		System.setProperty("spark.testing", "true")

		val conf = new SparkConf()
			.setMaster("local[2]")
			.setAppName("TensorEngineTest")
			.set("spark.ui.enabled", "false")
			.set("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
		sc = new SparkContext(conf)
		sc.setLogLevel("ERROR")
	}

	override protected def afterAll(): Unit = {
		if (sc != null) {
			sc.stop()
			sc = null
		}
		if (tempHadoopDir != null) {
			deleteRecursively(tempHadoopDir)
			tempHadoopDir = null
		}
		super.afterAll()
	}

	private def ensureWinutilsStub(): Unit = {
		val isWindows = System.getProperty("os.name").toLowerCase(java.util.Locale.ENGLISH).contains("win")
		if (!isWindows) return

		val alreadySet = Option(System.getProperty("hadoop.home.dir"))
			.exists(path => new File(path, "bin/winutils.exe").exists())
		if (alreadySet) return

		val tempDir = Files.createTempDirectory("spark-hadoop").toFile
		val binDir = new File(tempDir, "bin")
		if (!binDir.exists()) binDir.mkdirs()
		val winutils = new File(binDir, "winutils.exe")
		if (!winutils.exists() && !winutils.createNewFile()) {
			throw new IllegalStateException("Failed to create stub winutils.exe")
		}
		tempHadoopDir = tempDir
		System.setProperty("hadoop.home.dir", tempDir.getAbsolutePath)
	}

	private def deleteRecursively(file: File): Unit = {
		if (file.isDirectory) {
			file.listFiles().foreach(deleteRecursively)
		}
		file.delete()
	}

	private def sparseTensorFrom(entries: Seq[(Array[Int], Double)], shape: Array[Int]): SparseTensor = {
		SparseTensor(sc.parallelize(entries), shape)
	}

	private def denseMatrixFrom(rows: Array[Array[Double]]): DenseMatrix = {
		val indexed = rows.zipWithIndex.map { case (row, idx) => (idx, row.clone()) }
		DenseMatrix(sc.parallelize(indexed), rows.length, if (rows.isEmpty) 0 else rows.head.length)
	}

	private def naiveMttkrp(entries: Seq[(Array[Int], Double)],
												factors: Array[Array[Array[Double]]],
												targetMode: Int): Map[Int, Array[Double]] = {
		require(factors.nonEmpty, "At least one factor matrix required for naive check")
		val rank = factors.head.head.length
		val acc = scala.collection.mutable.Map.empty[Int, Array[Double]]

		entries.foreach { case (coords, value) =>
			val contrib = Array.fill(rank)(value)
			var mode = 0
			while (mode < factors.length) {
				if (mode != targetMode) {
					val row = factors(mode)(coords(mode))
					var r = 0
					while (r < rank) {
						contrib(r) *= row(r)
						r += 1
					}
				}
				mode += 1
			}

			val key = coords(targetMode)
			val bucket = acc.getOrElseUpdate(key, Array.fill(rank)(0.0))
			var r = 0
			while (r < rank) {
				bucket(r) += contrib(r)
				r += 1
			}
		}

		acc.iterator.map { case (k, arr) => k -> arr.clone() }.toMap
	}

	"TensorEngine.mttkrp" should "match naive 3-way computation" in {
		val entries = Seq(
			(Array(0, 0, 0), 1.0),
			(Array(1, 0, 1), 2.0),
			(Array(0, 1, 1), 3.0)
		)
		val shape = Array(2, 2, 2)
		val tensor = sparseTensorFrom(entries, shape)

		val factor0 = Array(Array(1.0, 0.5), Array(0.3, 1.2))
		val factor1 = Array(Array(1.1, 0.7), Array(0.8, 1.5))
		val factor2 = Array(Array(0.9, 1.3), Array(1.4, 0.6))
		val factorsData = Array(factor0, factor1, factor2)
		val factorMatrices = factorsData.map(denseMatrixFrom)

		val targetMode = 1
		val expected = naiveMttkrp(entries, factorsData, targetMode)
		val actual = TensorEngine
			.mttkrp(tensor, factorMatrices, targetMode)
			.collect()
			.toMap

		actual.keySet shouldBe expected.keySet
		actual.foreach { case (key, values) =>
			val expectedValues = expected(key)
			values should have length expectedValues.length
			values.zip(expectedValues).foreach { case (out, exp) =>
				out shouldBe (exp +- 1e-6)
			}
		}
	}

	it should "match naive computation when targeting mode 0" in {
		val entries = Seq(
			(Array(0, 0, 1), 4.0),
			(Array(1, 1, 0), 5.0)
		)
		val shape = Array(2, 2, 2)
		val tensor = sparseTensorFrom(entries, shape)

		val factor0 = Array(Array(0.6, 1.1), Array(1.5, 0.2))
		val factor1 = Array(Array(0.4, 0.9), Array(1.2, 1.7))
		val factor2 = Array(Array(1.0, 0.3), Array(0.8, 1.6))
		val factorsData = Array(factor0, factor1, factor2)
		val factorMatrices = factorsData.map(denseMatrixFrom)

		val expected = naiveMttkrp(entries, factorsData, targetMode = 0)
		val actual = TensorEngine
			.mttkrp(tensor, factorMatrices, targetMode = 0)
			.collect()
			.toMap

		actual should have size expected.size
		actual.foreach { case (key, values) =>
			values.zip(expected(key)).foreach { case (out, exp) =>
				out shouldBe (exp +- 1e-6)
			}
		}
	}

	it should "require a factor matrix per tensor mode" in {
		val tensor = sparseTensorFrom(Seq((Array(0, 0, 0), 1.0)), Array(1, 1, 1))
		val factor0 = denseMatrixFrom(Array(Array(1.0)))
		val factor1 = denseMatrixFrom(Array(Array(1.0)))

		val ex = intercept[IllegalArgumentException] {
			TensorEngine.mttkrp(tensor, Seq(factor0, factor1), targetMode = 0)
		}
		ex.getMessage should include ("Need one factor matrix per tensor mode")
	}

	it should "reject factor matrices with mismatched rank" in {
		val tensor = sparseTensorFrom(Seq((Array(0, 0, 0), 1.0)), Array(1, 1, 1))
		val factor0 = denseMatrixFrom(Array(Array(1.0, 2.0)))
		val factor1 = denseMatrixFrom(Array(Array(1.0, 2.0)))
		val factor2 = denseMatrixFrom(Array(Array(1.0, 2.0, 3.0)))

		val ex = intercept[IllegalArgumentException] {
			TensorEngine.mttkrp(tensor, Seq(factor0, factor1, factor2), targetMode = 0)
		}
		ex.getMessage should include ("must have the same column count")
	}

	it should "reject factor matrices whose row counts do not match the tensor shape" in {
		val tensor = sparseTensorFrom(Seq((Array(0, 0, 0), 1.0)), Array(2, 1, 1))
		val factor0 = denseMatrixFrom(Array(Array(1.0)))
		val factor1 = denseMatrixFrom(Array(Array(1.0)))
		val factor2 = denseMatrixFrom(Array(Array(1.0)))

		val ex = intercept[IllegalArgumentException] {
			TensorEngine.mttkrp(tensor, Seq(factor0, factor1, factor2), targetMode = 0)
		}
		ex.getMessage should include ("expected 2")
	}

	it should "reject target modes outside the tensor order" in {
		val tensor = sparseTensorFrom(Seq((Array(0, 0, 0), 1.0)), Array(1, 1, 1))
		val factor = denseMatrixFrom(Array(Array(1.0)))

		intercept[IllegalArgumentException] {
			TensorEngine.mttkrp(tensor, Seq(factor, factor, factor), targetMode = 3)
		}
	}

	it should "handle duplicate entries within a fiber" in {
        val entries = Seq(
            (Array(0, 0, 0), 1.0),
            (Array(0, 0, 0), 2.5),
            (Array(1, 1, 0), 3.0)
        )
        val tensor = sparseTensorFrom(entries, shape = Array(2, 2, 1))

        val factor0 = Array(Array(1.0, 0.5), Array(0.4, 1.2))
        val factor1 = Array(Array(1.0, 0.3), Array(0.9, 1.1))
        val factor2 = Array(Array(1.0, 1.0))
        val factors = Array(factor0, factor1, factor2).map(denseMatrixFrom)

        val expected = naiveMttkrp(entries, Array(factor0, factor1, factor2), targetMode = 0)
        val actual = TensorEngine.mttkrp(tensor, factors, targetMode = 0).collect().toMap

        actual.keySet shouldBe expected.keySet
        actual.foreach { case (idx, row) =>
            row.zip(expected(idx)).foreach { case (out, exp) => out shouldBe (exp +- 1e-6) }
        }
    }

	it should "handle skewed prefixes and empty partitions" in {
        val repeated = Seq.fill(50)((Array(0, 0, 0), 1.0))
        val sparse = Seq((Array(1, 1, 1), 2.0))
        val entries = (repeated ++ sparse).zipWithIndex.map { case (e, i) =>
            (e._1, e._2 * (1 + i % 3).toDouble)
         }.map { case (coords, value) => (coords, value) }

        val rdd = sc.parallelize(entries, numSlices = 16) // force many empty partitions
        val tensor = SparseTensor(rdd, shape = Array(2, 2, 2))

        val factor0 = Array(Array(0.6, 1.1, 0.2), Array(1.4, 0.5, 0.9))
        val factor1 = Array(Array(0.3, 0.8, 1.0), Array(1.2, 0.7, 0.4))
        val factor2 = Array(Array(0.5, 1.3, 0.6), Array(0.9, 0.2, 1.1))
        val factorsData = Array(factor0, factor1, factor2)
        val factorMatrices = factorsData.map(denseMatrixFrom)

        val expected = naiveMttkrp(entries, factorsData, targetMode = 2)
        val actual = TensorEngine.mttkrp(tensor, factorMatrices, targetMode = 2).collect().toMap

        actual.keySet shouldBe expected.keySet
        actual.foreach { case (idx, row) =>
            row.zip(expected(idx)).foreach { case (out, exp) => out shouldBe (exp +- 1e-5) }
        }
    }

	it should "support higher-order tensors" in {
		val entries = Seq(
			(Array(0, 0, 0, 0), 1.5),
			(Array(1, 1, 1, 2), 2.0),
			(Array(0, 1, 0, 1), 0.75),
			(Array(1, 0, 1, 0), 3.2)
		)
		val shape = Array(2, 2, 2, 3)
		val tensor = sparseTensorFrom(entries, shape)

		val factor0 = Array(
			Array(0.9, 0.1, 0.4),
			Array(1.2, 0.3, 0.7)
		)
		val factor1 = Array(
			Array(0.5, 0.8, 0.6),
			Array(1.1, 0.4, 0.2)
		)
		val factor2 = Array(
			Array(0.7, 0.6, 1.0),
			Array(0.9, 1.3, 0.5)
		)
		val factor3 = Array(
			Array(0.3, 0.4, 0.9),
			Array(1.4, 0.2, 0.1),
			Array(0.8, 1.1, 0.5)
		)
		val factorsData = Array(factor0, factor1, factor2, factor3)
		val factorMatrices = factorsData.map(denseMatrixFrom)

		val targetMode = 2
		val expected = naiveMttkrp(entries, factorsData, targetMode)
		val actual = TensorEngine.mttkrp(tensor, factorMatrices, targetMode).collect().toMap

		actual.keySet shouldBe expected.keySet
		actual.foreach { case (idx, row) =>
			row.zip(expected(idx)).foreach { case (out, exp) => out shouldBe (exp +- 1e-6) }
		}
	}

	it should "return empty output for empty tensors" in {
		val entries = Seq.empty[(Array[Int], Double)]
		val shape = Array(2, 3, 1)
		val tensor = sparseTensorFrom(entries, shape)

		val factor0 = Array(Array(1.0, 0.2, 0.8), Array(0.6, 0.4, 1.1))
		val factor1 = Array(Array(0.9, 1.3, 0.7), Array(0.5, 0.2, 0.6), Array(1.4, 0.1, 0.3))
		val factor2 = Array(Array(0.8, 0.9, 1.0))
		val factorMatrices = Array(factor0, factor1, factor2).map(denseMatrixFrom)

		val result = TensorEngine.mttkrp(tensor, factorMatrices, targetMode = 1).collect()
		result shouldBe empty
	}

	it should "produce the same output for COO and CSF implementations" in {
		val entries = Seq(
			(Array(0, 0, 0), 0.8),
			(Array(0, 1, 1), 1.1),
			(Array(1, 0, 1), 2.4),
			(Array(1, 1, 0), 3.3)
		)
		val tensor = sparseTensorFrom(entries, shape = Array(2, 2, 2))

		val factor0 = Array(Array(0.6, 1.0, 0.3), Array(0.9, 0.2, 1.1))
		val factor1 = Array(Array(1.2, 0.7, 0.5), Array(0.4, 0.8, 1.3))
		val factor2 = Array(Array(0.5, 1.4, 0.6), Array(1.0, 0.3, 0.9))
		val factorMatrices = Array(factor0, factor1, factor2).map(denseMatrixFrom)

		val targetMode = 1
		val csf = TensorEngine.mttkrp(tensor, factorMatrices, targetMode).collect().toMap
		val coo = TensorEngine.mttkrpCoo(tensor, factorMatrices, targetMode).collect().toMap

		csf.keySet shouldBe coo.keySet
		csf.foreach { case (idx, csfRow) =>
			val cooRow = coo(idx)
			csfRow.zip(cooRow).foreach { case (lhs, rhs) => lhs shouldBe (rhs +- 1e-6) }
		}
	}

	it should "integrate with LinearAlgebraAPI" in {
		val entries = Seq(
			(Array(0, 0, 0), 1.0),
			(Array(0, 1, 1), 0.5),
			(Array(1, 0, 1), 2.0)
		)
		val tensor = sparseTensorFrom(entries, shape = Array(2, 2, 2))

		val factor0 = Array(Array(0.6, 1.0, 0.3), Array(0.9, 0.2, 1.1))
		val factor1 = Array(Array(1.2, 0.7, 0.5), Array(0.4, 0.8, 1.3))
		val factor2 = Array(Array(0.5, 1.4, 0.6), Array(1.0, 0.3, 0.9))
		val factorsData = Array(factor0, factor1, factor2)
		val factorMatrices = factorsData.map(denseMatrixFrom)

		val expected = naiveMttkrp(entries, factorsData, targetMode = 1)
		val actual = LinearAlgebraAPI
			.mttkrp(tensor, factorMatrices, targetMode = 1)
			.rows
			.collect()
			.toMap

		actual.keySet shouldBe expected.keySet
		actual.foreach { case (idx, row) =>
			row.zip(expected(idx)).foreach { case (out, exp) => out shouldBe (exp +- 1e-6) }
		}
	}
}

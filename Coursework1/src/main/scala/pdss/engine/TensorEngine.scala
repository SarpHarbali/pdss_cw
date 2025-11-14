package pdss.engine

import org.apache.spark.HashPartitioner
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import pdss.core.{DenseMatrix, SparseTensor}
import scala.collection.mutable

object TensorEngine {


  def mttkrp(tensor: SparseTensor,
             factorMatrices: Seq[DenseMatrix],
             targetMode: Int): RDD[(Int, Array[Double])] = {
    require(factorMatrices.length == tensor.order, "Need one factor matrix per tensor mode")
    require(targetMode >= 0 && targetMode < tensor.order, s"Target mode $targetMode outside tensor order ${tensor.order}")

    val rank = factorMatrices.headOption.map(_.nCols.toInt).getOrElse {
      throw new IllegalArgumentException("At least one factor matrix is required")
    }

    factorMatrices.foreach { mat =>
      require(mat.nCols == rank, "All factor matrices must have the same column count (rank)")
    }

    factorMatrices.zipWithIndex.foreach { case (mat, mode) =>
      require(mat.nRows == tensor.shape(mode).toLong, s"Factor matrix for mode $mode has ${mat.nRows} rows, expected ${tensor.shape(mode)}")
    }

    val processModes = (0 until tensor.order).filter(_ != targetMode)
    val partitioner = new HashPartitioner(tensor.entries.context.defaultParallelism * 2)

    val entriesWithPrefix: RDD[(Vector[Int], (Int, Double))] = tensor.entries.map { case (indices, value) =>
      val prefix = processModes.map(indices).toVector
      val targetIndex = indices(targetMode)
      (prefix, (targetIndex, value))
    }

    val fibers: RDD[(Vector[Int], FiberData)] = entriesWithPrefix
      .combineByKey[mutable.HashMap[Int, Double]](
        (pair: (Int, Double)) => mutable.HashMap(pair._1 -> pair._2),
        (acc: mutable.HashMap[Int, Double], pair: (Int, Double)) => {
          val (idx, v) = pair
          acc.update(idx, acc.getOrElse(idx, 0.0) + v)
          acc
        },
        (left: mutable.HashMap[Int, Double], right: mutable.HashMap[Int, Double]) => {
          right.foreach { case (idx, v) => left.update(idx, left.getOrElse(idx, 0.0) + v) }
          left
        },
        partitioner
      )
      .map { case (prefix, map) =>
        val sorted: Seq[(Int, Double)] = map.toSeq.sortBy(_._1)
        val (indicesSeq, valuesSeq) = sorted.unzip
        (prefix, FiberData(indicesSeq.toArray, valuesSeq.toArray))
      }
      .partitionBy(partitioner)

    var fiberState: RDD[(Vector[Int], (FiberData, Array[Double]))] = fibers.map { case (prefix, data) =>
      (prefix, (data, Array.fill(rank)(1.0)))
    }

    processModes.zipWithIndex.foreach { case (mode, pos) =>
      val factorRows = factorMatrices(mode).rows.partitionBy(partitioner)
      fiberState = fiberState
        .map { case (prefix, (data, accum)) =>
          val idx = prefix(pos)
          (idx, (prefix, data, accum))
        }
        .join(factorRows)
        .map { case (_, ((prefix, data, accum), rowValues)) =>
          var r = 0
          while (r < accum.length) {
            accum(r) *= rowValues(r)
            r += 1
          }
          (prefix, (data, accum))
        }
        .partitionBy(partitioner)
    }

    fiberState
      .flatMap { case (_, (fiber, accum)) =>
        val coords = fiber.targetIndices
        val values = fiber.values
        val results = new Array[(Int, Array[Double])](coords.length)
        var i = 0
        while (i < coords.length) {
          val scaled = new Array[Double](accum.length)
          val scalar = values(i)
          var r = 0
          while (r < accum.length) {
            scaled(r) = accum(r) * scalar
            r += 1
          }
          results(i) = (coords(i), scaled)
          i += 1
        }
        results
      }
      .reduceByKey(partitioner, (left, right) => {
        var r = 0
        while (r < left.length) {
          left(r) += right(r)
          r += 1
        }
        left
      })
  }

  private case class FiberData(targetIndices: Array[Int], values: Array[Double]) {
    require(targetIndices.length == values.length, "Fiber indices and values must align")
  }
}

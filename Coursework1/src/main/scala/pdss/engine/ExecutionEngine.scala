package pdss.engine

import org.apache.spark.HashPartitioner
import org.apache.spark.rdd.RDD
import pdss.core.{DistVector, SparseMatrix, CSRMatrix}
import org.apache.spark.SparkContext._

import scala.reflect.ClassTag

object ExecutionEngine {

  // co-partition helper
  private def coPartition[V: ClassTag, W: ClassTag](
      left: RDD[(Int, V)],
      right: RDD[(Int, W)]
  ): (RDD[(Int, V)], RDD[(Int, W)]) = {
    val p = new HashPartitioner(left.sparkContext.defaultParallelism * 2)
    val L = left.partitionBy(p).persist()
    val R = right.partitionBy(p).persist()
    (L, R)
  }

  /** baseline SpMV (COO × vector) stays same */
  def spmv(A: SparseMatrix, x: DistVector): RDD[(Int, Double)] = {
    val AkeyedByJ: RDD[(Int, (Int, Double))] = A.entries.map { case (i, j, v) => (j, (i, v)) }
    val xByJ: RDD[(Int, Double)] = x.values

    val joined = AkeyedByJ.join(xByJ)
    val partials = joined.map { case (_, ((i, v), xj)) => (i, v * xj) }
    partials.reduceByKey(_ + _)
  }

  /** SpMM 1: COO × COO (senin mevcut halin) */
  def spmm(A: SparseMatrix, B: SparseMatrix): RDD[((Int, Int), Double)] = {
    val AbyK: RDD[(Int, (Int, Double))] = A.entries.map { case (i, k, vA) => (k, (i, vA)) }
    val BbyK: RDD[(Int, (Int, Double))] = B.entries.map { case (k, j, vB) => (k, (j, vB)) }

    val (acp, bcp) = coPartition(AbyK, BbyK)

    val joined = acp.join(bcp) // (k, ((i,vA),(j,vB)))
    val products = joined.map { case (_, ((i, vA), (j, vB))) =>
      ((i, j), vA * vB)
    }
    products.reduceByKey(_ + _)
  }

  /** SpMM 2: CSR × COO  */
  def spmmCSRWithCOO(A: CSRMatrix, B: SparseMatrix): RDD[((Int, Int), Double)] = {

    // expand CSR rows to (k,(i,vA))
    val AbyK: RDD[(Int, (Int, Double))] = A.rows.flatMap { row =>
      val i = row.row
      val cols = row.colIdx
      val vals = row.values
      val buf = new scala.collection.mutable.ArrayBuffer[(Int, (Int, Double))](cols.length)
      var t = 0
      while (t < cols.length) {
        buf += ((cols(t), (i, vals(t))))
        t += 1
      }
      buf
    }

    val BbyK: RDD[(Int, (Int, Double))] = B.entries.map { case (k, j, vB) => (k, (j, vB)) }

    val (acp, bcp) = coPartition(AbyK, BbyK)

    val joined = acp.join(bcp) // (k, ((i,vA),(j,vB)))
    val products = joined.map { case (_, ((i, vA), (j, vB))) =>
      ((i, j), vA * vB)
    }
    products.reduceByKey(_ + _)
  }
}

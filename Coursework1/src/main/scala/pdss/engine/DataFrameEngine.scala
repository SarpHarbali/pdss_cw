package pdss.engine

import org.apache.spark.HashPartitioner
import org.apache.spark.rdd.RDD
import pdss.core.{DistVector, SparseMatrix}
import org.apache.spark.SparkContext._

import scala.reflect.ClassTag

/** Baseline RDD engine using only map/flatMap/join/reduceByKey.
 * - No collect/broadcast of large data.
 * - No groupByKey (prefer reduceByKey with local combining).
 *
 * SpMV: y = A * x
 * SpMM: C = A * B   (both in COO)
 */
object ExecutionEngine {

  private def coPartition[V: ClassTag](            // <-- add ClassTag for V
                                                   left: RDD[(Int, V)],
                                                   right: RDD[(Int, V)]
                                      ): (RDD[(Int, V)], RDD[(Int, V)]) = {
    val p = new HashPartitioner(left.sparkContext.defaultParallelism * 2)
    val L = left.partitionBy(p).persist()
    val R = right.partitionBy(p).persist()
    (L, R)
  }

  /** SpMV (distributed): A (i,j,v) join x (j,xj) on j, then sum by row i. */
  def spmv(A: SparseMatrix, x: DistVector): RDD[(Int, Double)] = {
    val AkeyedByJ: RDD[(Int, (Int, Double))] = A.entries.map { case (i, j, v) => (j, (i, v)) }
    val xByJ: RDD[(Int, Double)] = x.values

    val joined = AkeyedByJ.join(xByJ)            // (j, ((i,v), xj))
    val partials = joined.map { case (_, ((i, v), xj)) => (i, v * xj) }
    partials.reduceByKey(_ + _)                  // y(i) = sum_j A(i,j)*x(j)
  }

  /** SpMM (distributed sparse × sparse COO):
   * A(i,k,vA) with B(k,j,vB) join on k → emit ((i,j), vA*vB) → sum.
   */
  def spmm(A: SparseMatrix, B: SparseMatrix): RDD[((Int, Int), Double)] = {
    val AbyK: RDD[(Int, (Int, Double))] = A.entries.map { case (i, k, vA) => (k, (i, vA)) }
    val BbyK: RDD[(Int, (Int, Double))] = B.entries.map { case (k, j, vB) => (k, (j, vB)) }
    val (acp, bcp) = coPartition(AbyK, BbyK)

    val joined = acp.join(bcp)                   // (k, ((i,vA),(j,vB)))
    val products = joined.map { case (_, ((i, vA), (j, vB))) => ((i, j), vA * vB) }
    products.reduceByKey(_ + _)                  // C(i,j) = sum_k A(i,k)*B(k,j)
  }
}

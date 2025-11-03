package pdss.engine

import org.apache.spark.HashPartitioner
import org.apache.spark.rdd.RDD
import pdss.core.{DistVector, SparseMatrix}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, sum}  // Add this import!



import scala.reflect.ClassTag

/** Baseline RDD engine using only map/flatMap/join/reduceByKey.
 * - No collect/broadcast of large data.
 * - No groupByKey (prefer reduceByKey with local combining).
 *
 * SpMV: y = A * x
 * SpMM: C = A * B   (both in COO)
 */
object DataFrameEngine {

  private def matrixToDF(A: SparseMatrix)(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._
    // A.entries is RDD[(Int, Int, Double)] in COO format
    A.entries
      .map { case (i, j, v) => (i, j, v) }
      .toDF("i", "j", "v")
  }



  /** SpMM (distributed sparse × sparse COO):
   * A(i,k,vA) with B(k,j,vB) join on k → emit ((i,j), vA*vB) → sum.
   */
  def spmm(A: SparseMatrix, B: SparseMatrix)(implicit spark: SparkSession): DataFrame = {
    val aDF = matrixToDF(A)  // Schema: (i, j, v) - we'll treat j as k
      .withColumnRenamed("j", "k")
      .withColumnRenamed("v", "v_a")

    val bDF = matrixToDF(B)  // Schema: (i, j, v) - we'll treat i as k
      .withColumnRenamed("i", "k")
      .withColumnRenamed("v", "v_b")

    // Join on k (middle dimension), multiply values, group by (i,j) and sum
    aDF.join(bDF, "k")
      .withColumn("product", col("v_a") * col("v_b"))
      .groupBy("i", "j")
      .agg(sum("product").alias("v"))
      .select("i", "j", "v")
      .orderBy("i", "j")
  }
}

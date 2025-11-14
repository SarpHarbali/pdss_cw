package pdss.engine

import org.apache.spark.HashPartitioner
import org.apache.spark.rdd.RDD
import pdss.core.{DistVector, SparseMatrix}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, sum}



import scala.reflect.ClassTag

object DataFrameEngine {

  private def matrixToDF(A: SparseMatrix)(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._
    A.entries
      .map { case (i, j, v) => (i, j, v) }
      .toDF("i", "j", "v")
  }



  def spmm(A: SparseMatrix, B: SparseMatrix)(implicit spark: SparkSession): DataFrame = {
    val aDF = matrixToDF(A)
      .withColumnRenamed("j", "k")
      .withColumnRenamed("v", "v_a")

    val bDF = matrixToDF(B)
      .withColumnRenamed("i", "k")
      .withColumnRenamed("v", "v_b")


    aDF.join(bDF, "k")
      .withColumn("product", col("v_a") * col("v_b"))
      .groupBy("i", "j")
      .agg(sum("product").alias("v"))
      .select("i", "j", "v")
      .orderBy("i", "j")
  }
}

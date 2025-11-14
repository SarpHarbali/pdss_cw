package pdss.core

import org.apache.spark.rdd.RDD

case class SparseMatrix(entries: RDD[(Int, Int, Double)],
                        nRows: Long,
                        nCols: Long)

case class DistVector(values: RDD[(Int, Double)], length: Long)

case class DenseMatrix(rows: RDD[(Int, Array[Double])],
                       nRows: Long,
                       nCols: Long)

case class CSRRow(row: Int, colIdx: Array[Int], values: Array[Double])

case class CSRMatrix(rows: RDD[CSRRow], nRows: Long, nCols: Long)

case class CSCCol(col: Int, rowIdx: Array[Int], values: Array[Double])
case class CSCMatrix(cols: RDD[CSCCol], nRows: Long, nCols: Long)

case class SparseTensor(entries: RDD[(Array[Int], Double)], shape: Array[Int]) {
    require(shape.nonEmpty, "Tensor must have at least one dimension")

    val order: Int = shape.length
}
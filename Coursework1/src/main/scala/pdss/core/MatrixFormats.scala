package pdss.core

import org.apache.spark.rdd.RDD

/** COO-style sparse matrix: entries are (i, j, v). */
case class SparseMatrix(entries: RDD[(Int, Int, Double)],
                        nRows: Long,
                        nCols: Long)

/** Distributed vector as (j, x_j) pairs. Works for dense or sparse. */
case class DistVector(values: RDD[(Int, Double)], length: Long)

case class DenseMatrix(rows: RDD[(Int, Array[Double])],
                       nRows: Long,
                       nCols: Long)

case class CSRRow(row: Int, colIdx: Array[Int], values: Array[Double])

case class CSRMatrix(rows: RDD[CSRRow], nRows: Long, nCols: Long)

case class CSCCol(col: Int, rowIdx: Array[Int], values: Array[Double])
case class CSCMatrix(cols: RDD[CSCCol], nRows: Long, nCols: Long)
package pdss.io

import org.apache.spark.SparkContext
import pdss.core.{SparseMatrix, DistVector}
import scala.util.Try

object Loader {

  /** Split on comma or tab */
  private def splitLine(line: String): Array[String] =
    line.split("[,\\t]")

  /** True if the string parses to Int */
  private def isInt(s: String): Boolean = Try(s.toInt).isSuccess
  /** True if the string parses to Double */
  private def isDouble(s: String): Boolean = Try(s.toDouble).isSuccess

  /** Load COO format from CSV/TSV.
   * Accepts lines like: i,j,v
   * - Skips header lines (e.g., "row,col,value")
   * - Skips comments starting with '#'
   * - Accepts comma or tab as delimiter
   */
  def loadCOO(sc: SparkContext, path: String): SparseMatrix = {
    val parsed = sc.textFile(path)
      .map(_.trim)
      .filter(l => l.nonEmpty && !l.startsWith("#"))
      .map(splitLine)
      // keep only valid triples (i,j,v)
      .filter(arr => arr.length >= 3 && isInt(arr(0)) && isInt(arr(1)) && isDouble(arr(2)))
      .map { arr =>
        (arr(0).toInt, arr(1).toInt, arr(2).toDouble)
      }

    // Dimensions via max indices (0-based)
    // NOTE: This is a small action, OK to infer dims.
    val maxI = parsed.map(_._1).max()
    val maxJ = parsed.map(_._2).max()
    SparseMatrix(parsed, nRows = maxI.toLong + 1, nCols = maxJ.toLong + 1)
  }

  def loadCSVToCOO(sc: SparkContext, path: String): SparseMatrix = {
    val lines = sc.textFile(path)
      .map(_.trim)
      .filter(l => l.nonEmpty && !l.startsWith("#"))
      .zipWithIndex()

    val cooTriples = lines.flatMap { case (line, rowIdx) =>
      val values = splitLine(line)
      values.zipWithIndex.flatMap { case (valueStr, colIdx) =>
        if (isDouble(valueStr)) {
          val value = valueStr.toDouble
          if (value != 0.0) {
            Some((rowIdx.toInt, colIdx, value))
          } else {
            None
          }
        } else {
          None
        }
      }
    }

    // Determine dimensions
    val nRows = lines.count()
    val nCols = lines.first()._1.split("[,\\t]").length.toLong

    SparseMatrix(cooTriples, nRows = nRows, nCols = nCols)
  }

  /** Load distributed vector j,val (or tab-separated).
   * Skips header lines like "col,value" or "index,value".
   */
  def loadVector(sc: SparkContext, path: String): DistVector = {
    val parsed = sc.textFile(path)
      .map(_.trim)
      .filter(l => l.nonEmpty && !l.startsWith("#"))
      .map(splitLine)
      .filter(arr => arr.length >= 2 && isInt(arr(0)) && isDouble(arr(1)))
      .map { arr =>
        (arr(0).toInt, arr(1).toDouble)
      }

    val len = parsed.map(_._1).max().toLong + 1
    DistVector(parsed, length = len)
  }
}

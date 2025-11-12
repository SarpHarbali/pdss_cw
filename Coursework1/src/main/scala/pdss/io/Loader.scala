package pdss.io

import org.apache.spark.SparkContext
import pdss.core.{SparseMatrix, DistVector, CSRMatrix, CSRRow, CSCMatrix, CSCCol, DenseMatrix, SparseTensor}
import scala.util.Try

object Loader {

  /** Split on comma or tab */
  private def splitLine(line: String): Array[String] =
    line.split("[,\\t]")

  private def isInt(s: String): Boolean = Try(s.toInt).isSuccess
  private def isDouble(s: String): Boolean = Try(s.toDouble).isSuccess

  /** Load COO format from CSV/TSV: i,j,v */
  def loadCOO(sc: SparkContext, path: String): SparseMatrix = {
    val parsed = sc.textFile(path)
      .map(_.trim)
      .filter(l => l.nonEmpty && !l.startsWith("#"))
      .map(splitLine)
      .filter(arr => arr.length >= 3 && isInt(arr(0)) && isInt(arr(1)) && isDouble(arr(2)))
      .map { arr =>
        (arr(0).toInt, arr(1).toInt, arr(2).toDouble)
      }

    val maxI = parsed.map(_._1).max()
    val maxJ = parsed.map(_._2).max()
    SparseMatrix(parsed, nRows = maxI.toLong + 1, nCols = maxJ.toLong + 1)
  }

  /** Dense-like CSV → COO */
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
          if (value != 0.0) Some((rowIdx.toInt, colIdx, value)) else None
        } else None
      }
    }

    val nRows = lines.count()
    val nCols = lines.first()._1.split("[,\\t]").length.toLong

    SparseMatrix(cooTriples, nRows = nRows, nCols = nCols)
  }

  /** index,value */
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

  /** Convert COO (SparseMatrix) to CSRMatrix (row-grouped) */
  def cooToCSR(m: SparseMatrix): CSRMatrix = {
    val rows = m.entries
      .groupBy(_._1) // group by row i
      .map { case (i, triples) =>
        val arr = triples.map { case (_, j, v) => (j, v) }.toArray
        val sorted = arr.sortBy(_._1) // nicer to keep cols ordered
        val colIdx = sorted.map(_._1)
        val values = sorted.map(_._2)
        CSRRow(i, colIdx, values)
      }

    CSRMatrix(rows, m.nRows, m.nCols)
  }

    /** Convert COO (i,j,v) to CSC (col-grouped) */
  def cooToCSC(m: SparseMatrix): CSCMatrix = {
    val cols = m.entries
      .groupBy(_._2) // group by column j
      .map { case (j, triples) =>
        // triples: Iterable[(i, j, v)]
        val arr = triples.map { case (i, _, v) => (i, v) }.toArray
        val sorted = arr.sortBy(_._1) // sort by row
        val rowIdx = sorted.map(_._1)
        val values = sorted.map(_._2)
        CSCCol(j, rowIdx, values)
      }

    CSCMatrix(cols, m.nRows, m.nCols)
  }

  def loadDenseMatrixRows(sc: SparkContext, path: String): DenseMatrix = {
    val lines = sc.textFile(path)
      .map(_.trim)
      .filter(_.nonEmpty)
      .zipWithIndex()

    val rows = lines.map { case (line, rowIdx) =>
      val parts = line.split(",")
      val arr = new Array[Double](parts.length)
      var k = 0
      while (k < parts.length) {
        arr(k) = parts(k).toDouble
        k += 1
      }
      (rowIdx.toInt, arr)
    }

    val numRows = lines.count()
    val numCols = rows.first()._2.length.toLong
    DenseMatrix(rows, numRows, numCols)
  }
  //////////////////
  // Tensor shit ///
  //////////////////
  private val shapeRegex = "(?i)^#\\s*shape\\s*:(.*)$".r

  /** Load an N-way sparse tensor stored as `[i1,i2,...,in,value]` per line with a `# Shape:` header. THANK YOU PIAZZA*/
  def loadSparseTensor(sc: SparkContext, path: String): SparseTensor = {
    val raw = sc.textFile(path)
      .map(_.trim)
      .filter(_.nonEmpty)

    val shapeLine = raw
      .filter(line => shapeRegex.pattern.matcher(line).matches())
      .take(1)
      .headOption
      .getOrElse(throw new IllegalArgumentException(s"Missing `# Shape:` header in $path"))

    val dims = shapeLine match {
      case shapeRegex(rest) =>
        rest.split("[,\\s]+").filter(_.nonEmpty).map(_.toInt)
      case _ => throw new IllegalStateException("Unexpected shape header format")
    }

    val order = dims.length
    require(order > 0, "Tensor shape must include at least one dimension")
    val dimsLocal = dims

    val entries = raw
      .filter(line => !line.startsWith("#"))
      .map { line =>
        val parts = line.split("[,\\s]+").filter(_.nonEmpty)
        if (parts.length != order + 1) {
          throw new IllegalArgumentException(s"Tensor entry `$line` does not have $order indices plus value")
        }

        val idx = Array.ofDim[Int](order)
        var m = 0
        while (m < order) {
          if (!isInt(parts(m))) {
            throw new IllegalArgumentException(s"Index `${parts(m)}` is not an integer in `$line`")
          }
          val asInt = parts(m).toInt
          if (asInt < 0 || asInt >= dimsLocal(m)) {
            throw new IllegalArgumentException(s"Index $asInt out of bounds for mode $m with size ${dimsLocal(m)}")
          }
          idx(m) = asInt
          m += 1
        }

        val valueStr = parts.last
        if (!isDouble(valueStr)) {
          throw new IllegalArgumentException(s"Value `$valueStr` is not numeric in `$line`")
        }
        val cell = valueStr.toDouble
        (idx, cell)
      }
      .filter { case (_, value) => value != 0.0 }

    SparseTensor(entries, dims)
  }

}

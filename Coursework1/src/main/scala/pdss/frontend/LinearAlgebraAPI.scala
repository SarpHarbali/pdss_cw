package pdss.frontend

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import pdss.core._
import pdss.engine.{ExecutionEngine, TensorEngine}

object LinearAlgebraAPI {

  sealed trait SparseFormat
  object SparseFormat {
    case object COO extends SparseFormat
    case object CSR extends SparseFormat
    case object CSC extends SparseFormat
    case object CSR_CSC extends SparseFormat
  }

  def spmv(
      A: SparseMatrix,
      x: DistVector,
      useCSR: Boolean
  ): DistVector = {
    val result: RDD[(Int, Double)] =
      if (useCSR) {
        val csr = pdss.io.Loader.cooToCSR(A)
        ExecutionEngine.spmvCSR(csr, x)
      } else {
        ExecutionEngine.spmv(A, x)
      }

    DistVector(result, length = A.nRows)
  }

  def spmv(
      A: SparseMatrix,
      x: Array[Double],
      useCSR: Boolean
  )(implicit sc: SparkContext): DistVector = {
    val xBroadcast: Broadcast[Array[Double]] = sc.broadcast(x)

    val result: RDD[(Int, Double)] =
      if (useCSR) {
        val csr = pdss.io.Loader.cooToCSR(A)
        ExecutionEngine.spmvCsrWithDense(csr, xBroadcast)
      } else {
        ExecutionEngine.spmvCooWithDense(A, xBroadcast)
      }

    DistVector(result, length = A.nRows)
  }

  def spmm(
      A: SparseMatrix,
      B: SparseMatrix,
      format: SparseFormat
  ): SparseMatrix = {
    require(
      A.nCols == B.nRows,
      s"Dimension mismatch in SpMM: A is ${A.nRows}×${A.nCols}, B is ${B.nRows}×${B.nCols}"
    )

    val resultPairs: RDD[((Int, Int), Double)] = format match {
      case SparseFormat.COO =>
        ExecutionEngine.spmm(A, B)
      case SparseFormat.CSR =>
        val csrA = pdss.io.Loader.cooToCSR(A)
        val csrB = pdss.io.Loader.cooToCSR(B)
        ExecutionEngine.spmmCSR(csrA, csrB)
      case SparseFormat.CSC =>
        val cscA = pdss.io.Loader.cooToCSC(A)
        val cscB = pdss.io.Loader.cooToCSC(B)
        ExecutionEngine.spmmCSC(cscA, cscB)
      case SparseFormat.CSR_CSC =>
        val csrA = pdss.io.Loader.cooToCSR(A)
        val cscB = pdss.io.Loader.cooToCSC(B)
        ExecutionEngine.spmmCSRWithCSC(csrA, cscB)
    }

    val entries: RDD[(Int, Int, Double)] =
      resultPairs.map { case ((i, j), v) => (i, j, v) }

    SparseMatrix(entries, nRows = A.nRows, nCols = B.nCols)
  }

  def spmm(
      A: SparseMatrix,
      B: DenseMatrix
  ): DenseMatrix = {
    require(
      A.nCols == B.nRows,
      s"Dimension mismatch in SpMM (sparse × dense): A is ${A.nRows}×${A.nCols}, B is ${B.nRows}×${B.nCols}"
    )

    val resultRows: RDD[(Int, Array[Double])] =
      ExecutionEngine.spmm_dense(A, B)

    DenseMatrix(
      rows = resultRows,
      nRows = A.nRows,
      nCols = B.nCols
    )
  }

  def spmmChain3(
      A: SparseMatrix,
      B: SparseMatrix,
      C: SparseMatrix,
      format: SparseFormat
  ): SparseMatrix = {
    val plan = pdss.engine.ChainPlanner.chooseOrder3(A, B, C)
    plan.order match {
      case "AB_then_C" =>
        val AB = spmm(A, B, format)
        spmm(AB, C, format)
      case "A_then_BC" =>
        val BC = spmm(B, C, format)
        spmm(A, BC, format)
      case other =>
        throw new IllegalArgumentException(s"Unknown plan: $other")
    }
  }

  def mttkrp(
      tensor: SparseTensor,
      factorMats: Seq[DenseMatrix],
      targetMode: Int
  ): DenseMatrix = {
    val rows: RDD[(Int, Array[Double])] =
      TensorEngine.mttkrp(tensor, factorMats, targetMode)

    val rank: Long =
      factorMats.headOption.map(_.nCols).getOrElse {
        throw new IllegalArgumentException("Need at least one factor matrix to determine rank.")
      }

    val nRows: Long = tensor.shape(targetMode).toLong

    DenseMatrix(
      rows = rows,
      nRows = nRows,
      nCols = rank
    )
  }
}

package pdss.frontend

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import pdss.core._
import pdss.engine.{ExecutionEngine, TensorEngine}

/**
 * ============================================
 * User-facing Linear Algebra & Tensor Frontend
 * ============================================
 *
 * Public API used by "end users" of your library / coursework.
 * They never see CSR/CSC, broadcast details, or join tricks.
 *
 * Supported operations (matching the rubric table):
 *
 * SpMV:
 *   - SparseMatrix × DistVector         (sparse vector)
 *   - SparseMatrix × Array[Double]      (dense local vector)
 *
 * SpMM:
 *   - SparseMatrix × SparseMatrix       (COO / CSR / CSC internally)
 *   - SparseMatrix × DenseMatrix        (dense right operand)
 *
 * Tensor algebra:
 *   - MTTKRP using SparseTensor + DenseMatrix factors
 *     via pdss.engine.TensorEngine.
 */
object LinearAlgebraAPI {

  // -----------------------------
  // Internal format selector
  // -----------------------------
  sealed trait SparseFormat
  object SparseFormat {
    case object COO extends SparseFormat
    case object CSR extends SparseFormat
    case object CSC extends SparseFormat
  }

  // ============================================================
  // 1) SpMV FRONTEND  (NO default arguments)
  // ============================================================

  /**
   * SpMV with a sparse / distributed vector (DistVector).
   * useCSR = true   → convert A to CSR and use CSR kernel
   * useCSR = false  → stay in COO and use COO kernel
   */
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

  /**
   * SpMV with a dense local vector (Array[Double]), broadcast internally.
   * useCSR = true   → convert A to CSR and use CSR+dense kernel
   * useCSR = false  → stay in COO and use COO+dense kernel
   */
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

  // ============================================================
  // 2) SpMM FRONTEND: Sparse × Sparse
  // ============================================================

  /**
   * SpMM: C = A * B   (both sparse).
   *
   * Rubric line:
   *   Operation: SpMM
   *   Left:      Sparse Matrix
   *   Right:     Sparse Matrix
   */
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
        // Baseline COO × COO
        ExecutionEngine.spmm(A, B)

      case SparseFormat.CSR =>
        // Convert both to CSR and use CSR × CSR implementation
        val csrA = pdss.io.Loader.cooToCSR(A)
        val csrB = pdss.io.Loader.cooToCSR(B)
        ExecutionEngine.spmmCSR(csrA, csrB)

      case SparseFormat.CSC =>
        // Convert both to CSC and use CSC × CSC implementation
        val cscA = pdss.io.Loader.cooToCSC(A)
        val cscB = pdss.io.Loader.cooToCSC(B)
        ExecutionEngine.spmmCSC(cscA, cscB)
    }

    val entries: RDD[(Int, Int, Double)] =
      resultPairs.map { case ((i, j), v) => (i, j, v) }

    SparseMatrix(entries, nRows = A.nRows, nCols = B.nCols)
  }

  // ============================================================
  // 3) SpMM FRONTEND: Sparse × Dense
  // ============================================================

  /**
   * SpMM: C = A * B, where A is sparse (COO) and B is a dense row-wise matrix.
   *
   * Rubric line:
   *   Operation: SpMM
   *   Left:      Sparse Matrix   (COO)
   *   Right:     Dense Matrix    (DenseMatrix)
   */
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
      rows  = resultRows,
      nRows = A.nRows,
      nCols = B.nCols
    )
  }

  // ============================================================
  // 4) Optional: chain of 3 sparse matrices (uses ChainPlanner)
  // ============================================================

  /**
   * Example: multiply three sparse matrices with a simple cost-based plan.
   * This is just a nice extra that shows how your frontend can orchestrate
   * multiple SpMM calls.
   */
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

  // ============================================================
  // 5) Tensor Algebra FRONTEND (MTTKRP)
  // ============================================================

  /**
   * Frontend entry for MTTKRP, built on top of your TensorEngine.
   *
   * - tensor:       SparseTensor from pdss.core (COO-style, with .order and .shape)
   * - factorMats:   one DenseMatrix per mode of the tensor
   * - targetMode:   which mode we are computing MTTKRP in (0-based)
   *
   * Returns:
   *   A DenseMatrix of size (tensor.shape(targetMode) × rank),
   *   where rank = number of columns in the factor matrices.
   *
   * This API is what you describe in the "tensor algebra beyond SpMV/SpMM"
   * part of the report. Implementation details live in TensorEngine.
   */
  def mttkrp(
      tensor: SparseTensor,
      factorMats: Seq[DenseMatrix],
      targetMode: Int
  ): DenseMatrix = {
    // Delegate the heavy lifting to TensorEngine
    val rows: RDD[(Int, Array[Double])] =
      TensorEngine.mttkrp(tensor, factorMats, targetMode)

    val rank: Long =
      factorMats.headOption.map(_.nCols).getOrElse {
        throw new IllegalArgumentException("Need at least one factor matrix to determine rank.")
      }

    val nRows: Long = tensor.shape(targetMode).toLong

    DenseMatrix(
      rows  = rows,
      nRows = nRows,
      nCols = rank
    )
  }
}

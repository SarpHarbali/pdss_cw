package pdss.engine

import pdss.core._
object ChainPlanner {

  /** Simple result class you can use in Main */
  case class Plan3(order: String, firstCost: Double, secondCost: Double)

  private case class SimpleStats(nRows: Long, nCols: Long, nnz: Long)

  private def stats(m: SparseMatrix): SimpleStats = {
    val nnz = m.entries.count()
    SimpleStats(m.nRows, m.nCols, nnz)
  }

  /** Very simple sparse join-cost proxy for A(i,k) × B(k,j). */
  private def estPairs(A: SimpleStats, B: SimpleStats): Double = {
    val k = math.min(A.nCols, B.nRows).toDouble
    val avgDegA = if (A.nCols == 0) 0.0 else A.nnz.toDouble / A.nCols.toDouble   // nnz per col of A
    val avgDegB = if (B.nRows == 0) 0.0 else B.nnz.toDouble / B.nRows.toDouble   // nnz per row of B
    avgDegA * avgDegB * k
  }

  /** Decide between (A·B)·C vs A·(B·C). */
  def chooseOrder3(A: SparseMatrix, B: SparseMatrix, C: SparseMatrix): Plan3 = {
    val sA = stats(A)
    val sB = stats(B)
    val sC = stats(C)

    val costAB = estPairs(sA, sB)
    val costBC = estPairs(sB, sC)

    if (costAB <= costBC) {
      val sAB = SimpleStats(sA.nRows, sB.nCols, nnz = math.round(costAB))
      val thenC = estPairs(sAB, sC)
      Plan3("AB_then_C", costAB, thenC)
    } else {
      val sBC = SimpleStats(sB.nRows, sC.nCols, nnz = math.round(costBC))
      val thenA = estPairs(sA, sBC)
      Plan3("A_then_BC", costBC, thenA)
    }
  }
}

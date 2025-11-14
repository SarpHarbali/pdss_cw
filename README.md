1. Overview

This project implements a distributed linear–algebra and tensor–algebra engine using Apache Spark.
The system exposes a clean user-facing frontend API that supports:

SpMV (Sparse Matrix × Vector)

SpMM (Sparse Matrix × Matrix)

Sparse × Dense operations

Multiple sparse formats (COO, CSR, CSC)

Tensor algebra (MTTKRP)

Users interact with a small, simple API and never need to interact with Spark, RDDs, or storage formats.
All low-level logic (joins, partitioning, CSR/CSC kernels) is encapsulated in the execution engine.

2. Project Structure
pdss/
 ├── core/              # Data structures (SparseMatrix, DenseMatrix, DistVector, SparseTensor)
 ├── engine/            # ExecutionEngine + SpMV/SpMM/MTTKRP kernels
 ├── frontend/          # LinearAlgebraAPI: user-facing interface
 ├── io/                # Loader for CSV/COO/tensor inputs
 ├── tests/             # FrontendTests
 └── Main.scala         # Optional demo (not required for marking)

Responsibilities

Loader: Handles all file loading and conversion into core types.

Frontend: Exposes clean operations (spmv, spmm, mttkrp).

Engine: Implements low-level Spark kernels.

Core: Defines all matrix/tensor representations.

3. Build Instructions
Requirements

Scala 2.12

sbt 1.x

Spark 3.x (local mode supported)

Compile
sbt compile

Run tests
sbt test


All the operations are done via:

import pdss.frontend.LinearAlgebraAPI frontend


Example:

val C = LinearAlgebraAPI.spmm(A, B)
val y = LinearAlgebraAPI.spmv(A, x)
val M = LinearAlgebraAPI.mttkrp(tensor, factors, mode)


5. Frontend API Summary
Sparse Format Selector
sealed trait SparseFormat
object SparseFormat { case object COO; case object CSR; case object CSC }

SpMV
def spmv(A: SparseMatrix, x: DistVector, useCSR: Boolean = false): DistVector
def spmv(A: SparseMatrix, x: Array[Double], useCSR: Boolean = false)
        (implicit sc: SparkContext): DistVector

SpMM
def spmm(A: SparseMatrix, B: SparseMatrix,
         format: SparseFormat = COO): SparseMatrix

def spmm(A: SparseMatrix, B: DenseMatrix): DenseMatrix

Chained Sparse Operations
def spmmChain3(A: SparseMatrix, B: SparseMatrix, C: SparseMatrix,
               format: SparseFormat = COO): SparseMatrix

Tensor Algebra
def mttkrp(tensor: SparseTensor,
           factorMats: Seq[DenseMatrix],
           targetMode: Int): DenseMatrix

6. Loader (Input Data)

The Loader supports:

Dense CSV → DenseMatrix

COO triples CSV → SparseMatrix

Vector CSV → DistVector

COO tensor files → SparseTensor

Example:

val A = Loader.loadSparseMatrixCOO(sc, "A.csv")
val x = Loader.loadVector(sc, "x.csv")
val tensor = Loader.loadSparseTensor(sc, "tensor.csv")

Frontend never performs file I/O; only the Loader does.

7. Execution Engine (Internal Kernels)

The engine implements:

COO-based SpMV

CSR-based SpMV

COO-based SpMM

CSR × CSR sparse–sparse SpMM

CSC × CSC sparse–sparse SpMM

CSR × CSC hybrid kernels

Sparse × Dense SpMM

MTTKRP (generic Khatri–Rao mode-wise formulation)

Conversions (COO↔CSR/CSC) are internal and hidden from users.

8. Correctness Guarantees & Assumptions

No collect() is used in frontend or engine kernels.

All operations are lazy Spark transformations.

Frontend is purely an API layer (no I/O).

Loader supports fully dense files (zeros included).

Execution engine is deterministic and partition-safe.

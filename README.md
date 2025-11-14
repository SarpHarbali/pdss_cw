PDSS Coursework — Distributed Linear & Tensor Algebra Engine
1. Overview

This project implements a distributed linear-algebra and tensor-algebra engine using Apache Spark.
The system exposes a user-facing frontend API that supports:

SpMV (Sparse Matrix × Vector)

SpMM (Sparse Matrix × Matrix)

Sparse × Dense operations

Multiple sparse storage formats: COO, CSR, CSC

Tensor algebra: MTTKRP

Execution planning (e.g., 3-matrix chain multiplication)

Users interact only with a small, clean API.
They never deal with:

Spark RDDs

Partitioners

CSR/CSC internal kernels

Broadcasts or joins

All low-level logic lives in the ExecutionEngine.

2. Project Structure
pdss/
├── core/            # Data structures: SparseMatrix, DenseMatrix, DistVector, SparseTensor
│
├── engine/          # ExecutionEngine: SpMV, SpMM (COO/CSR/CSC), the dense kernel, MTTKRP
│
├── frontend/        # LinearAlgebraAPI: user-facing operations (spmv, spmm, mttkrp)
│
├── io/              # Loader: CSV → COO, vectors, tensors, etc.
│
├── tests/           # FrontendTests: correctness tests for all operations
│
└── Main.scala       # Optional demo runner (not required for marking)

3. Component Responsibilities
Loader

Reads input CSV/dense files (including zeros)

Converts them to internal structures:

SparseMatrix (COO)

DenseMatrix

DistVector

SparseTensor (COO-style)

ExecutionEngine

Implements all distributed kernels:

SpMV (COO)

SpMV (CSR)

SpMV (COO × Dense)

SpMV (CSR × Dense)

SpMM (COO × COO)

SpMM (CSR × CSR)

SpMM (CSC × CSC)

CSR × CSC (hybrid multiplication)

SpMM (Sparse × Dense)

Tensor algebra MTTKRP

Handles partitioning, co-partitioning, joins, and broadcast logic.

Frontend (LinearAlgebraAPI)

Provides clean API functions:

spmv(A, x)
spmm(A, B)
spmm(A, B_dense)
spmmChain3(A, B, C)
mttkrp(tensor, factors, targetMode)


Converts COO → CSR/CSC based on user selection.

Hides all RDD and kernel details.

Tests (FrontendTests)

Validate:

SpMV sparse vs dense paths

SpMM COO/CSR/CSC kernel agreement

Sparse × Dense SpMM correctness

CSV loader correctness

Tensor MTTKRP output correctness

Main.scala

Optional example runner (not required by markers).

4. Running the Project
Prerequisites

Scala 2.12

sbt 1.x

Apache Spark 3.x

Java 11 or 17

To compile:
sbt compile

To run frontend tests:
sbt "runMain pdss.FrontendTests"

To run the demo main (optional):
sbt "runMain pdss.Main"

5. Notes for Markers (Important)

No collect() is used inside the frontend API

Only test code uses collect() for correctness checking

Frontend does not do file I/O

Loader handles all CSV → structures

ExecutionEngine contains all join/broadcast/format logic

Frontend is intentionally thin, following coursework instructions

6. License

This project was developed for the University of Edinburgh PDSS coursework.

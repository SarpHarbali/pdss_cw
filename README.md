# PDSS Coursework 1

Distributed sparse linear algebra and tensor engine implemented on Apache Spark. The project exposes a user-facing API (`pdss.frontend.LinearAlgebraAPI`) and a distributed execution layer (`pdss.engine.ExecutionEngine` / `TensorEngine`) that cover:

- Sparse Matrix–Vector multiplication (SpMV) for both dense and sparse vectors, and COO / CSR storage.
- Sparse Matrix–Matrix multiplication (SpMM) across COO, CSR, CSC and mixed formats plus sparse–dense products.
- Tensor algebra via MTTKRP (Matricized Tensor Times Khatri-Rao Product) on multi-way sparse tensors.

## Requirements

- Java 17 (Adoptium Temurin 17.0.x recommended).
- sbt 1.11.x (auto-configured via `project/build.properties`).
- Spark dependencies are pulled automatically (`spark-core`, `spark-sql`, `spark-mllib` 3.0.3).
- Windows users should ensure `winutils.exe` is on the path or let the tests create the temporary stub.

## Build & Test

```powershell
cd Coursework1
sbt test
```

Useful focused commands:

```powershell
cd Coursework1
sbt "testOnly pdss.TensorEngineTest"     # tensor algebra suite
sbt "testOnly pdss.FrontendTests"        # API-level smoke tests
sbt run                                   # executes pdss.Main (SpMM chain benchmark)
```

All tests run locally with Spark `local[*]`. `TensorEngineTest` spins up its own `SparkContext`, creates synthetic tensors, and compares MTTKRP outputs to a naive CPU baseline.

## Project Layout

- `src/main/scala/pdss/core`: core data structures (`SparseMatrix`, `DenseMatrix`, `SparseTensor`, CSR/CSC helpers).
- `src/main/scala/pdss/engine`: execution engines for matrices (`ExecutionEngine`), tensors (`TensorEngine`), and planners/benchmarks.
- `src/main/scala/pdss/frontend`: `LinearAlgebraAPI` – user-facing operations wrapping the engines.
- `src/test/scala/pdss`: ScalaTest suites including `TensorEngineTest`.
- `data/`: CSV inputs produced by `DatasetGen` (COO triplets and vectors) used by demos/benchmarks.

## Data & Input Formats

| Structure        | Expected shape | Backing RDD payload |
|------------------|----------------|---------------------|
| `SparseMatrix`   | `nRows × nCols` | `RDD[(rowIdx: Int, colIdx: Int, value: Double)]` (COO) |
| `DenseMatrix`    | `nRows × nCols` | `RDD[(rowIdx: Int, rowValues: Array[Double])]` |
| `DistVector`     | `length`        | `RDD[(index: Int, value: Double)]` |
| `SparseTensor`   | `shape: Array[Int]` (one entry per mode) | `RDD[(coords: Array[Int], value: Double)]` |

**Tensor inputs** must always carry the full `shape`, even if the tensor has zero entries. Every factor matrix supplied to `TensorEngine.mttkrp` must:

1. Correspond to one tensor mode (same ordering as `shape`).
2. Have `nRows == shape(mode)`.
3. Share the same number of columns (the rank `R`).

The target mode is a 0-based index selecting which mode’s rows the output spans. These constraints are enforced via `require(...)` statements, so mis-specified inputs fail fast with descriptive errors.

## Tensor Algebra Example

```scala
import org.apache.spark.{SparkConf, SparkContext}
import pdss.core.{DenseMatrix, SparseTensor}
import pdss.frontend.LinearAlgebraAPI

val conf = new SparkConf().setMaster("local[2]").setAppName("MTTKRP-demo")
val sc   = new SparkContext(conf)

val tensorEntries = Seq(
  (Array(0, 0, 0), 1.0),
  (Array(0, 1, 1), 0.5),
  (Array(1, 0, 1), 2.0)
)
val tensor = SparseTensor(sc.parallelize(tensorEntries), shape = Array(2, 2, 2))

val factorRows = Seq(
  Array(Array(0.6, 1.0), Array(0.9, 0.2)),  // mode-0
  Array(Array(1.2, 0.7), Array(0.4, 0.8)),  // mode-1
  Array(Array(0.5, 1.4), Array(1.0, 0.3))   // mode-2
)
val factors = factorRows.map { rows =>
  val indexed = rows.zipWithIndex.map { case (row, idx) => (idx, row) }
  DenseMatrix(sc.parallelize(indexed), rows.length, rows.head.length)
}

val result = LinearAlgebraAPI.mttkrp(tensor, factors, targetMode = 1)
result.rows.collect().foreach { case (modeIdx, values) =>
  println(s"row=$modeIdx -> ${values.mkString(", ")}")
}

sc.stop()
```

`LinearAlgebraAPI.mttkrp` returns a `DenseMatrix` whose rows correspond to the target mode; the code above collects it only because the example tensor is tiny. Avoid `collect()` for real workloads.

## Matrix Operations from the Frontend

```scala
import pdss.frontend.LinearAlgebraAPI
import pdss.frontend.LinearAlgebraAPI.SparseFormat._

val A: SparseMatrix = Loader.loadCSVToCOO(sc, "data/sparse_matrix_normal.csv")
val xDense: Array[Double] = Array(2.0, 0.0, 1.0)
val y = LinearAlgebraAPI.spmv(A, xDense, useCSR = true)(sc)

val B: SparseMatrix = Loader.loadCSVToCOO(sc, "data/sparse_matrix_normal2.csv")
val C = LinearAlgebraAPI.spmm(A, B, format = CSR)
val D = LinearAlgebraAPI.spmm(A, DenseMatrix(...))                  // sparse × dense
```

Internally these calls map to the execution-engine methods and stay entirely on RDDs; no large `collect()` is performed.

## Sample Benchmarks & Utilities

- `pdss.Main`: demonstrates SpMM chain planning (`ChainPlanner`) and runtime comparison of multiplication orders.
- `pdss.PartitionBenchmark`, `pdss.SpmmBenchmark`, `pdss.SpmmScaleBenchmark`, `pdss.SpmvBenchmarkFromData`: benchmarking utilities to study partitioning, scaling, and format trade-offs.
- `pdss.TensorMttkrpBenchmark`: compares COO streaming vs fiber-compressed (CSF) MTTKRP implementations on synthetic tensors and writes `tensor_mttkrp_results.csv`.
- `pdss.DatasetGen`: generates random COO matrices/vectors in `data/` for experiments. Run via `sbt "runMain pdss.DatasetGen"`.

Each benchmark has its own `main` method; invoke with `sbt "runMain <fully.qualified.Object>"`.

### Tensor Benchmark Usage

```powershell
cd Coursework1
sbt "runMain pdss.TensorMttkrpBenchmark"
```

Optional overrides can be provided as CLI flags, e.g. `--shape=256,256,256 --density=0.0003 --rank=48 --target=1 --trials=5`. Results list per-trial runtimes for both the naive COO path (`TensorEngine.mttkrpCoo`) and the CSF-based implementation (`TensorEngine.mttkrp`) so you can report speedups directly from the generated CSV.

## Troubleshooting

- **Missing winutils on Windows**: The tensor tests auto-create a temporary stub under `%TEMP%/spark-hadoop...`. You can also manually set `HADOOP_HOME` to a folder containing `bin/winutils.exe`.
- **Out-of-memory**: Adjust the `javaOptions` and `Test / javaOptions` in `build.sbt` (current defaults: `-Xms4G`, `-Xmx8G`).
- **Dataset paths**: Benchmarks expect files under `data/`. Run `DatasetGen` or adjust the paths in the corresponding `Main` objects.

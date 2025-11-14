# PDSS Project: A Distributed Engine for Large-Scale Sparse Matrix and Tensor Algebra

This project is a high-performance, distributed execution engine for large-scale sparse linear algebra operations (SpMV, SpMM) and tensor algebra (MTTKRP), built on Apache Spark's RDD primitives.

The engine optimizes memory usage by leveraging the nature of sparse data and parallelizes computations across a cluster using Spark's distributed `join` and `reduceByKey` operations.

## ✨ Key Features

* **SpMV Support:** Sparse Matrix x (Dense Vector or Sparse Vector) multiplication.
* **SpMM Support:** Sparse Matrix x (Dense Matrix or Sparse Matrix) multiplication.
* **Tensor Support:** Mode-n MTTKRP (Matricized Tensor Times Khatri-ao Product) for sparse tensors.
* **Data Ingestion:** Flexible data loading from COO (coordinate triplets), dense CSVs, and vector files.
* **Optimizations:**
    * Use of custom `HashPartitioner` for data co-location.
    * Internal support for CSR (Compressed Sparse Row) / CSC (Compressed Sparse Column) formats for efficient computation.
    * A simple cost-based optimizer for chained matrix multiplications via the `spmmChain3` method.
    * Smart caching (`persist()`) for RDDs that are reused.

---

## ⚙️ Tech Stack

* **Language:** Scala (v2.12.20)
* **Framework:** Apache Spark (v3.0.3)
* **Platform:** Java 11 (OpenJDK)
* **Build Tool:** `sbt` (Scala Build Tool) is recommended.

---

## 🚀 Setup and Getting Started

Follow these steps to build and run the project on your local machine.

### Prerequisites

1.  **Java 11 (or higher):**
    * Ensure you have JDK 11 installed.
    * You can check the version with `java -version`.
2.  **Apache Spark 3.0.3:**
    * Download `spark-3.0.3-bin-hadoop2.7` from the [official Spark archive](https://archive.apache.org/dist/spark/spark-3.0.3/).
    * Extract the archive to a directory (e.g., `/opt/spark/`).
    * Add the `SPARK_HOME` environment variable to your `.bashrc` or `.zshrc`:
        ```bash
        export SPARK_HOME=/path/to/spark-3.0.3-bin-hadoop2.7
        export PATH=$SPARK_HOME/bin:$PATH
        ```
3.  **sbt (Scala Build Tool):**
    * `sbt` is required to compile Scala projects. Follow the installation instructions on the [official website](https://www.scala-sbt.org/download.html).

### Building the Project

1.  Clone the repository:
    ```bash
    git clone <your-project-github-link>
    cd pdss-project
    ```

2.  Use `sbt` to compile the project and create a `.jar` package:
    ```bash
    sbt clean package
    ```
    This command will download dependencies (spark-core, spark-sql, etc.) and package your project into a JAR file under the `target/scala-2.12/` directory.

---

## 💻 Usage and Execution

The main interface for the project is the `LinearAlgebraAPI` object. Computations are triggered by calling methods on this object.

### API Usage Example

Here is an example of how to use the engine in your own Scala application:

```scala
import org.apache.spark.sql.SparkSession
import com.pdss.project.LinearAlgebraAPI // (Update this with your package name)
import com.pdss.project.Loader // (Path to your Loader object)

object MainDriver {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("PDSS Spark Algebra")
      .master("local[4]") // 4 cores for local testing
      .getOrCreate()
    
    implicit val sc = spark.sparkContext

    // 1. Load Data (using your Loader module)
    // Hypothetical file paths:
    val A = Loader.loadSparseMatrix("data/matrix_A.coo")
    val B = Loader.loadSparseMatrix("data/matrix_B.coo")
    val x_dense = Loader.loadDenseVectorAsArray("data/vector_x.csv")
    val x_dist = Loader.loadDistVector("data/vector_x_dist.coo")

    // --- API Usage ---

    // 2. SpMV (Sparse Matrix x Dense Vector)
    // The dense vector (Array[Double]) is broadcast.
    println("Running SpMV (Sparse x Dense)...")
    val y1 = LinearAlgebraAPI.spmv(A, x_dense)
    y1.take(10).foreach(println)

    // 3. SpMV (Sparse Matrix x Sparse Vector)
    // The distributed vector (DistVector) is used with an RDD join.
    println("Running SpMV (Sparse x Sparse)...")
    val y2 = LinearAlgebraAPI.spmv(A, x_dist)
    y2.take(10).foreach(println)

    // 4. SpMM (Sparse Matrix x Sparse Matrix)
    // Uses the default COO-based join.
    println("Running SpMM (Sparse x Sparse)...")
    val C1 = LinearAlgebraAPI.spmm(A, B)
    println(s"Result C1 NNZ: ${C1.entries.count()}")

    // 5. SpMM (Using optimized CSR/CSC)
    println("Running SpMM (Sparse x Sparse) with CSR/CSC...")
    val C2 = LinearAlgebraAPI.spmm(A, B, format = LinearAlgebraAPI.SparseFormat.CSR)
    println(s"Result C2 NNZ: ${C2.entries.count()}")
    
    // 6. Chain Multiplication
    // Selects the most efficient path (A(BC) or (AB)C).
    println("Running SpMM Chain (A*B*C)...")
    val C3 = Loader.loadSparseMatrix("data/matrix_C.coo")
    val D = LinearAlgebraAPI.spmmChain3(A, B, C3)
    println(s"Result D NNZ: ${D.entries.count()}")

    spark.stop()
  }
}

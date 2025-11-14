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
    git clone git@github.com:SarpHarbali/pdss_cw.git
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

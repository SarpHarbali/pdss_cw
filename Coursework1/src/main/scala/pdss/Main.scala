package pdss



import org.apache.spark.{SparkConf, SparkContext}
import pdss.engine.ExecutionEngine
import pdss.io.Loader

object Main extends App{
  private val conf: SparkConf = new SparkConf()
    .setAppName("ImdbAnalysis") // Set your application's name
    .setMaster("local[*]") // Use all cores of the local machine
    .set("spark.ui.enabled", "false")


  val sc: SparkContext = new SparkContext(conf)

  val A = Loader.loadCSVToCOO(sc, "src/main/data/sparse_matrix_normal.csv")      // rows: i,j,v


  // val x = Loader.loadVector(sc, "src/main/data/vector_x.csv")   // rows: j,val

  // --- SpMV (distributed join; no collect)
  // val y = ExecutionEngine.spmv(A, x)

  // --- Example SpMM using two sparse matrices (COO on both)
  val B = Loader.loadCSVToCOO(sc, "src/main/data/sparse_matrix_normal2.csv")
  val C = ExecutionEngine.spmm(A, B)




  println("C = A * B (first 10 nonzeros):")
  C.take(10).foreach { case ((i, j), v) => println(s"($i,$j) -> $v") }

  C.map { case ((i, j), v) => s"$i,$j,$v" }
    .saveAsTextFile("results/spmm_output")

  println("✅ Results written to results/spmv_output/ and results/spmm_output/")

  sc.stop()
}

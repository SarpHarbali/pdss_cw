import sbt.util

name := "Coursework1"
scalaVersion := "2.12.20"

scalacOptions ++= Seq("-deprecation")

// Use Maven Central for dependencies
resolvers += Resolver.mavenCentral

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.0.3",
  "org.apache.spark" %% "spark-sql" % "3.0.3",
  "junit" % "junit" % "4.10" % Test,
  "org.scalatest" %% "scalatest" % "3.2.9" % Test
)

logLevel := util.Level.Error

// Always fork JVM for 'run' task
fork := true

// Force stdout output and connect stdin
outputStrategy := Some(StdoutOutput)
connectInput := true

// JVM options for both test and run
javaOptions ++= Seq(
  "-Xms4G", 
  "-Xmx8G",
  // Required for Spark on Java 9+
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  "--add-opens=java.base/java.util=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED"
)

// Ensure test fork uses same options
Test / fork := true
Test / javaOptions ++= javaOptions.value

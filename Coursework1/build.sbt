import sbt.util

name := "Coursework1"
scalaVersion := "2.12.20"

scalacOptions ++= Seq("-deprecation")
resolvers ++= Resolver.sonatypeOssRepos("releases")

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.0.3",
  "junit" % "junit" % "4.10" % Test,
  "org.scalatest" %% "scalatest" % "3.2.9" % Test
)

logLevel := util.Level.Error

fork := false  // Don't fork a separate JVM
outputStrategy := Some(StdoutOutput)  // Force stdout output
connectInput := true  // Connect stdin

javaOptions ++= Seq(
  "-Xms4G",
  "-Xmx8G"
)

Compile / run / fork := true

Compile / run / javaOptions ++= Seq(
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  "--add-opens=java.base/java.util=ALL-UNNAMED"
)

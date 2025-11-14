import sbt.util

name := "Coursework1"
scalaVersion := "2.12.20"

scalacOptions ++= Seq("-deprecation")

resolvers += Resolver.mavenCentral

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.0.3",
  "org.apache.spark" %% "spark-sql" % "3.0.3",
  "junit" % "junit" % "4.10" % Test,
  "org.apache.spark" %% "spark-mllib" % "3.0.3",
  "org.scalatest" %% "scalatest" % "3.2.9" % Test
)

logLevel := util.Level.Error

fork := true

outputStrategy := Some(StdoutOutput)
connectInput := true

javaOptions ++= Seq(
  "-Xms4G", 
  "-Xmx8G",
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  "--add-opens=java.base/java.util=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED"
)

Test / fork := true
Test / javaOptions ++= javaOptions.value

Compile / run / fork := true

Compile / run / javaOptions ++= Seq(
  "--add-opens", "java.base/java.lang.invoke=ALL-UNNAMED"
)


plugins {
    id("java")
    id("application")
    id("me.champeau.jmh") version "0.7.2"
}

application {
    mainClass.set("com.tngtech.java_virtual_thread_benchmark.quicksort.QuickSortRunner")
}

jmh {
    val benchmark = findProperty("benchmark") as String?
    if (benchmark != null) {
        includes = listOf(benchmark)
    }

    iterations = 3
    warmupIterations = 1
    threads = 1

    jvmArgs.addAll(listOf(
        "-Xms4g", "-Xmx4g",
        "-Xlog:gc*:file=gc.log:time,level,tags"
    ))

    profilers = listOf("gc", "stack")
    resultFormat = "JSON"
    resultsFile = project.file("build/results/jmh/results.json")
}

group = "com.tngtech"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    compileOnly("org.projectlombok:lombok:1.18.40")
    annotationProcessor("org.projectlombok:lombok:1.18.40")

    testCompileOnly("org.projectlombok:lombok:1.18.40")
    testAnnotationProcessor("org.projectlombok:lombok:1.18.40")

    testImplementation(platform("org.junit:junit-bom:5.9.1"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")

    implementation("org.postgresql:postgresql:42.5.1")
    implementation("com.zaxxer:HikariCP:5.1.0")

    implementation("org.openjdk.jmh:jmh-core:1.37")
    implementation("org.openjdk.jmh:jmh-generator-annprocess:1.37")

    implementation("io.projectreactor:reactor-core:3.4.10")

    implementation("io.r2dbc:r2dbc-spi:1.0.0.RELEASE")
    implementation("io.r2dbc:r2dbc-pool:1.0.1.RELEASE")
    implementation("org.postgresql:r2dbc-postgresql:1.0.4.RELEASE")

    implementation("org.hibernate.orm:hibernate-core:6.4.4.Final")
    implementation("org.hibernate.orm:hibernate-hikaricp:6.4.4.Final")
}

tasks.withType<JavaCompile> {
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<JavaExec> {
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

# Benchmark Audit and Suggestions

## 1. Bug Report: Build Configuration (Java 25)

**Issue:**
The `build.gradle.kts` file specifies a toolchain version of `25`:
```kotlin
java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}
```
Java 25 is not yet a generally available standard release supported by most environments/Gradle toolchains (as of the current context). This causes the build to fail immediately on standard environments (like the one used for this audit, which has Java 21).

**Impact:**
The project cannot be built or tested without manual intervention.

**Suggestion:**
Downgrade the requirement to a stable LTS version like Java 21, or ensure the environment supports Java 25.
```kotlin
languageVersion.set(JavaLanguageVersion.of(21))
```
Additionally, the `test` task failed due to missing `junit-platform-launcher`. This should be added to dependencies:
```kotlin
testRuntimeOnly("org.junit.platform:junit-platform-launcher")
```

## 2. Bug Report: Platform Threads Ignore `maxDepth`

**Issue:**
In `QuickSortDepthBenchmark`, the `@Param("maxDepth")` parameter is intended to control the recursion depth for the sorting algorithm. However, this parameter is **only** used by the `ThreadMode.VIRTUAL` implementation. The `ThreadMode.PLATFORM` implementation (which uses `ForkJoinPool` and `RecursiveAction`) **ignores** `maxDepth` and instead relies solely on the `sequentialThreshold`.

**Impact:**
This invalidates the comparison between Platform and Virtual threads for the "depth" benchmark. The `platformThread` benchmark runs the exact same code (full recursion until threshold) for all 12 values of `maxDepth`, producing redundant and misleading data. You cannot see how "depth-limited" platform threads compare to "depth-limited" virtual threads because the platform version is never depth-limited.

**Suggestion:**
Modify the `QuickSortTask` (Platform implementation) to accept and respect a `maxDepth` parameter, similar to the Virtual Thread version.

**Proposed Logic:**
1.  Add a `depth` field to `QuickSortTask`.
2.  Increment `depth` in the constructor of sub-tasks.
3.  In `compute()`, check `if (depth > maxDepth || high - low < sequentialThreshold)`.
4.  If true, call `sequentialQuickSort`.

## 3. Experiment Design: Reproducibility and Seeding

**Observation:**
The `DataGenerator` class currently uses a hardcoded seed (`private static final long SEED = 42L`). This ensures that every run of the benchmark uses the *exact same* random array. This is generally good for comparing algorithms (control variable).

**Suggestion:**
However, relying on a hardcoded constant limits your ability to test the algorithm's robustness across *different* random distributions.
I recommend adding a `@Param("seed")` to the benchmark classes.
1.  Update `DataGenerator.generate` to accept a `long seed`.
2.  In `QuickSortBenchmark` and `QuickSortDepthBenchmark`, add:
    ```java
    @Param({"42"}) // can add more seeds like "123", "999" later
    private long seed;
    ```
3.  Pass this seed to `DataGenerator.generate`.

This allows you to verify that performance is consistent across different random datasets without recompiling.

## 4. Stress Test Stability (Memory & Saturation)

**Observation:**
`QuickSortDepthBenchmark` uses `SIZE = 100_000_000` integers (approx. 400MB).
With `maxDepth` set to 20, the Virtual Thread implementation will attempt to spawn approximately $2^{21}$ (2 million) virtual threads.
While Virtual Threads are lightweight, 2 million thread objects + closures + the 400MB array might approach the 2GB heap limit (`-Xmx2g`).

**Suggestion:**
1.  **Monitor GC**: Ensure you enable the GC profiler (already done via `profilers = listOf("gc")`) to check for high allocation rates or "Stop-The-World" pauses.
2.  **Increase Heap**: If you encounter `OutOfMemoryError` or extreme GC churn, consider increasing the heap to 4GB (`-Xmx4g`) in `build.gradle.kts`.
3.  **Saturation Point**: For 100M elements, the theoretical saturation point for parallelism (where tasks become too small, < 512 elements) is around depth $\log_2(100,000,000 / 512) \approx 17.6$.
    *   Your benchmark goes up to depth 20, which is good for testing the "oversubscription" or "overhead" behavior.
    *   Expect performance to degrade or flatten after depth 18.

## 5. Metrics and Profiling

**Observation:**
You are currently using the `gc` profiler. This provides allocation rates (`gc.alloc.rate`) and churn.

**Suggestion:**
For a more comprehensive analysis of "Heap Usage" and "Latency", consider these additional profilers:
1.  **Java Flight Recorder (JFR)**:
    *   Add `"-prof", "jfr"` to the profilers list (or command line).
    *   This generates a `.jfr` file that you can open in JDK Mission Control to see exact heap usage over time, thread states, and GC events.
    *   *Note*: This requires the commercial features unlock on some older JDKs, but is standard in OpenJDK 11+.
2.  **Stack Profiler**:
    *   Add `"-prof", "stack"` to see where the CPU time is spent. This helps verify if the overhead is coming from the sorting logic or the thread management (scheduling/parking).
3.  **PerfAsm (Linux only)**:
    *   If running on Linux, `"-prof", "perfasm"` gives assembly-level insight, showing if virtual threads incur more cache misses or instruction overhead.

## Summary of Recommended Actions

1.  **Fix Build**: Downgrade to Java 21 and add test runtime dependency.
2.  **Refactor `ParallelQuickSort`**: Update `QuickSortTask` to track recursion depth.
3.  **Update `DataGenerator`**: Allow passing a custom seed.
4.  **Update Benchmarks**: Add `@Param("seed")` and pass it down.
5.  **Configuration**: Consider bumping heap to 4GB for the 100M element test.
6.  **Profiling**: Add `jfr` or `stack` profiler for deeper insights.

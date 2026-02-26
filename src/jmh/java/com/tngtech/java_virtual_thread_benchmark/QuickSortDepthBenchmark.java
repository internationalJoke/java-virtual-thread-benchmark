package com.tngtech.java_virtual_thread_benchmark;

import com.tngtech.java_virtual_thread_benchmark.quicksort.DataGenerator;
import com.tngtech.java_virtual_thread_benchmark.quicksort.DataGenerator.Distribution;
import com.tngtech.java_virtual_thread_benchmark.quicksort.ParallelQuickSort;
import com.tngtech.java_virtual_thread_benchmark.quicksort.ParallelQuickSort.ThreadMode;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;

@BenchmarkMode({Mode.Throughput, Mode.AverageTime, Mode.SampleTime})
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Warmup(iterations = 2, time = 3)
@Measurement(iterations = 5, time = 5)
@Fork(value = 2, jvmArgs = {"-Xms4g", "-Xmx4g", "-Xlog:gc*:file=gc.log:time,level,tags"})
public class QuickSortDepthBenchmark {

    private static final int SIZE = 100_000_000;
    private static final int SEQUENTIAL_THRESHOLD = 512;

    @Param({"0", "2", "4", "6", "8", "10", "12", "14", "16", "18", "20", "22", "24", "26", "28", "30"})
    private int maxDepth;

    private int[] baseArray;
    private int[] workArray;

    @Setup(Level.Trial)
    public void generateData() {
        baseArray = DataGenerator.generate(Distribution.RANDOM, SIZE);
        workArray = new int[SIZE];
    }

    @Setup(Level.Invocation)
    public void copyArray() {
        System.arraycopy(baseArray, 0, workArray, 0, SIZE);
    }

    @Benchmark
    public void virtualThread(Blackhole bh) {
        ParallelQuickSort.sort(workArray, ThreadMode.VIRTUAL, maxDepth, SEQUENTIAL_THRESHOLD);
        bh.consume(workArray[workArray.length - 1]);
    }

    @Benchmark
    public void platformThread(Blackhole bh) {
        ParallelQuickSort.sort(workArray, ThreadMode.PLATFORM, maxDepth, SEQUENTIAL_THRESHOLD);
        bh.consume(workArray[workArray.length - 1]);
    }
}

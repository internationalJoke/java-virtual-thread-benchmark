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
@Warmup(iterations = 3, time = 5)
@Measurement(iterations = 5, time = 5)
@Fork(value = 2, jvmArgs = {"-Xms2g", "-Xmx2g"})
public class QuickSortBenchmark {

    @Param({"100000", "200000", "400000", "800000"})
    private int size;

    @Param({"RANDOM", "SORTED", "REVERSE_SORTED", "MANY_DUPLICATES"})
    private String distribution;

    private int[] baseArray;
    private int[] workArray;

    @Setup(Level.Trial)
    public void generateData() {
        Distribution dist = Distribution.valueOf(distribution);
        baseArray = DataGenerator.generate(dist, size);
    }

    @Setup(Level.Invocation)
    public void copyArray() {
        workArray = Arrays.copyOf(baseArray, baseArray.length);
    }

    @Benchmark
    public void platformThread(Blackhole bh) {
        ParallelQuickSort.sort(workArray, ThreadMode.PLATFORM);
        bh.consume(workArray[workArray.length - 1]);
    }

    @Benchmark
    public void virtualThread(Blackhole bh) {
        ParallelQuickSort.sort(workArray, ThreadMode.VIRTUAL);
        bh.consume(workArray[workArray.length - 1]);
    }
}

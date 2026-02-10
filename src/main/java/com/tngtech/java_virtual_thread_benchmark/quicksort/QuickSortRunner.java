package com.tngtech.java_virtual_thread_benchmark.quicksort;

import com.tngtech.java_virtual_thread_benchmark.quicksort.DataGenerator.Distribution;
import com.tngtech.java_virtual_thread_benchmark.quicksort.ParallelQuickSort.ThreadMode;

import java.util.Arrays;

/**
 * Standalone runner for quick ad-hoc comparisons.
 * Usage: ./gradlew run --args="100000"
 *        ./gradlew run --args="100000 200000 400000 800000"
 */
public class QuickSortRunner {

    private static final int WARMUP_ROUNDS = 3;
    private static final int MEASURE_ROUNDS = 5;

    public static void main(String[] args) {
        int[] sizes;
        if (args.length == 0) {
            sizes = new int[]{100_000, 200_000, 400_000, 800_000};
        } else {
            sizes = Arrays.stream(args).mapToInt(Integer::parseInt).toArray();
        }

        System.out.println("=".repeat(90));
        System.out.printf("%-18s | %-8s | %-12s | %-12s | %-10s%n",
                "Distribution", "Size", "Platform(ms)", "Virtual(ms)", "Ratio(V/P)");
        System.out.println("=".repeat(90));

        for (Distribution dist : Distribution.values()) {
            for (int size : sizes) {
                int[] baseArray = DataGenerator.generate(dist, size);

                // Warmup
                for (int w = 0; w < WARMUP_ROUNDS; w++) {
                    int[] copy = Arrays.copyOf(baseArray, baseArray.length);
                    ParallelQuickSort.sort(copy, ThreadMode.PLATFORM);
                    copy = Arrays.copyOf(baseArray, baseArray.length);
                    ParallelQuickSort.sort(copy, ThreadMode.VIRTUAL);
                }

                // Measure platform threads
                long[] platformTimes = new long[MEASURE_ROUNDS];
                for (int i = 0; i < MEASURE_ROUNDS; i++) {
                    int[] copy = Arrays.copyOf(baseArray, baseArray.length);
                    long start = System.nanoTime();
                    ParallelQuickSort.sort(copy, ThreadMode.PLATFORM);
                    platformTimes[i] = System.nanoTime() - start;
                    verifySorted(copy);
                }

                // Measure virtual threads
                long[] virtualTimes = new long[MEASURE_ROUNDS];
                for (int i = 0; i < MEASURE_ROUNDS; i++) {
                    int[] copy = Arrays.copyOf(baseArray, baseArray.length);
                    long start = System.nanoTime();
                    ParallelQuickSort.sort(copy, ThreadMode.VIRTUAL);
                    virtualTimes[i] = System.nanoTime() - start;
                    verifySorted(copy);
                }

                double platformMedianMs = medianNanos(platformTimes) / 1_000_000.0;
                double virtualMedianMs = medianNanos(virtualTimes) / 1_000_000.0;
                double ratio = virtualMedianMs / platformMedianMs;

                System.out.printf("%-18s | %8d | %12.2f | %12.2f | %10.2f%n",
                        dist, size, platformMedianMs, virtualMedianMs, ratio);
            }
        }
        System.out.println("=".repeat(90));

        System.out.println("\nLatency percentiles (last size per distribution):");
        System.out.println("-".repeat(90));

        for (Distribution dist : Distribution.values()) {
            int size = sizes[sizes.length - 1];
            int[] baseArray = DataGenerator.generate(dist, size);

            // Warmup
            for (int w = 0; w < WARMUP_ROUNDS; w++) {
                ParallelQuickSort.sort(Arrays.copyOf(baseArray, baseArray.length), ThreadMode.PLATFORM);
                ParallelQuickSort.sort(Arrays.copyOf(baseArray, baseArray.length), ThreadMode.VIRTUAL);
            }

            int runs = 20;
            long[] pTimes = new long[runs];
            long[] vTimes = new long[runs];
            for (int i = 0; i < runs; i++) {
                int[] copy = Arrays.copyOf(baseArray, baseArray.length);
                long s = System.nanoTime();
                ParallelQuickSort.sort(copy, ThreadMode.PLATFORM);
                pTimes[i] = System.nanoTime() - s;

                copy = Arrays.copyOf(baseArray, baseArray.length);
                s = System.nanoTime();
                ParallelQuickSort.sort(copy, ThreadMode.VIRTUAL);
                vTimes[i] = System.nanoTime() - s;
            }

            Arrays.sort(pTimes);
            Arrays.sort(vTimes);

            System.out.printf("%s (n=%d):%n", dist, size);
            System.out.printf("  Platform  -> p50=%.2fms  p95=%.2fms  p99=%.2fms%n",
                    pTimes[percentileIndex(runs, 50)] / 1e6,
                    pTimes[percentileIndex(runs, 95)] / 1e6,
                    pTimes[percentileIndex(runs, 99)] / 1e6);
            System.out.printf("  Virtual   -> p50=%.2fms  p95=%.2fms  p99=%.2fms%n",
                    vTimes[percentileIndex(runs, 50)] / 1e6,
                    vTimes[percentileIndex(runs, 95)] / 1e6,
                    vTimes[percentileIndex(runs, 99)] / 1e6);
        }
    }

    private static double medianNanos(long[] times) {
        long[] sorted = Arrays.copyOf(times, times.length);
        Arrays.sort(sorted);
        int mid = sorted.length / 2;
        if (sorted.length % 2 == 0) {
            return (sorted[mid - 1] + sorted[mid]) / 2.0;
        }
        return sorted[mid];
    }

    private static int percentileIndex(int count, int percentile) {
        int idx = (int) Math.ceil(percentile / 100.0 * count) - 1;
        return Math.min(idx, count - 1);
    }

    private static void verifySorted(int[] arr) {
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] < arr[i - 1]) {
                throw new AssertionError("Array not sorted at index " + i);
            }
        }
    }
}

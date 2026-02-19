package com.tngtech.java_virtual_thread_benchmark.quicksort;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class ParallelQuickSort {

    private static final int DEFAULT_SEQUENTIAL_THRESHOLD = 4096;
    private static final int DEFAULT_MAX_VIRTUAL_DEPTH = 16;

    public enum ThreadMode {
        PLATFORM,
        VIRTUAL
    }

    private ParallelQuickSort() {
    }

    public static void sort(int[] array, ThreadMode mode) {
        sort(array, mode, DEFAULT_MAX_VIRTUAL_DEPTH, DEFAULT_SEQUENTIAL_THRESHOLD);
    }

    public static void sort(int[] array, ThreadMode mode, int maxDepth, int sequentialThreshold) {
        if (array == null || array.length <= 1) return;
        int effectiveThreshold = Math.max(1, sequentialThreshold);

        if (mode == ThreadMode.PLATFORM) {
            ForkJoinPool.commonPool().invoke(new QuickSortTask(array, 0, array.length - 1, effectiveThreshold, 0, maxDepth));
        } else {
            virtualQuickSort(array, 0, array.length - 1, 0, maxDepth, effectiveThreshold);
        }
    }

    // ==================== Platform Thread version  ====================

    private static class QuickSortTask extends RecursiveAction {
        private final int[] array;
        private final int low;
        private final int high;
        private final int sequentialThreshold;
        private final int depth;
        private final int maxDepth;

        QuickSortTask(int[] array, int low, int high, int sequentialThreshold, int depth, int maxDepth) {
            this.array = array;
            this.low = low;
            this.high = high;
            this.sequentialThreshold = sequentialThreshold;
            this.depth = depth;
            this.maxDepth = maxDepth;
        }

        @Override
        protected void compute() {
            if (low >= high) return;

            if (high - low < sequentialThreshold || depth > maxDepth) {
                sequentialQuickSort(array, low, high);
                return;
            }

            int[] bounds = threeWayPartition(array, low, high);

            QuickSortTask left = new QuickSortTask(array, low, bounds[0] - 1, sequentialThreshold, depth + 1, maxDepth);
            QuickSortTask right = new QuickSortTask(array, bounds[1] + 1, high, sequentialThreshold, depth + 1, maxDepth);

            left.fork();
            right.compute();
            left.join();
        }
    }

    // ==================== Virtual Thread version ====================

    private static void virtualQuickSort(int[] array, int low, int high, int depth, int maxDepth, int sequentialThreshold) {
        if (low >= high) return;

        if (high - low < sequentialThreshold || depth > maxDepth) {
            sequentialQuickSort(array, low, high);
            return;
        }

        int[] bounds = threeWayPartition(array, low, high);
        int lt = bounds[0];
        int gt = bounds[1];

        Thread leftThread = Thread.ofVirtual().start(() ->
                virtualQuickSort(array, low, lt - 1, depth + 1, maxDepth, sequentialThreshold));

        virtualQuickSort(array, gt + 1, high, depth + 1, maxDepth, sequentialThreshold);

        try {
            leftThread.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
    }

    // ==================== Shared utilities ====================

    /**
     * Three-way partition (Dutch National Flag).
     * Returns [lt, gt] where:
     *   array[low..lt-1]  < pivot
     *   array[lt..gt]    == pivot
     *   array[gt+1..high] > pivot
     */
    static int[] threeWayPartition(int[] array, int low, int high) {
        // Median-of-three pivot selection
        int mid = low + (high - low) / 2;
        if (array[mid] < array[low]) swap(array, low, mid);
        if (array[high] < array[low]) swap(array, low, high);
        if (array[mid] < array[high]) swap(array, mid, high);
        int pivot = array[high];

        int lt = low;   // array[low..lt-1] < pivot
        int gt = high;  // array[gt+1..high] > pivot
        int i = low;

        while (i <= gt) {
            if (array[i] < pivot) {
                swap(array, lt, i);
                lt++;
                i++;
            } else if (array[i] > pivot) {
                swap(array, i, gt);
                gt--;
            } else {
                i++;
            }
        }

        return new int[]{lt, gt};
    }

    static void sequentialQuickSort(int[] array, int low, int high) {
        if (low >= high) return;

        if (high - low < 16) {
            insertionSort(array, low, high);
            return;
        }

        int[] bounds = threeWayPartition(array, low, high);
        sequentialQuickSort(array, low, bounds[0] - 1);
        sequentialQuickSort(array, bounds[1] + 1, high);
    }

    private static void insertionSort(int[] array, int low, int high) {
        for (int i = low + 1; i <= high; i++) {
            int key = array[i];
            int j = i - 1;
            while (j >= low && array[j] > key) {
                array[j + 1] = array[j];
                j--;
            }
            array[j + 1] = key;
        }
    }

    private static void swap(int[] array, int i, int j) {
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

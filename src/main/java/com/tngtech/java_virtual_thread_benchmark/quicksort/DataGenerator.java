package com.tngtech.java_virtual_thread_benchmark.quicksort;

import java.util.Random;

public class DataGenerator {

    public enum Distribution {
        RANDOM,
        SORTED,
        REVERSE_SORTED,
        MANY_DUPLICATES
    }

    private static final long SEED = 42L;

    public static int[] generate(Distribution distribution, int size) {
        return switch (distribution) {
            case RANDOM -> randomArray(size);
            case SORTED -> sortedArray(size);
            case REVERSE_SORTED -> reverseSortedArray(size);
            case MANY_DUPLICATES -> manyDuplicatesArray(size);
        };
    }

    private static int[] randomArray(int size) {
        Random rng = new Random(SEED);
        int[] arr = new int[size];
        for (int i = 0; i < size; i++) {
            arr[i] = rng.nextInt();
        }
        return arr;
    }

    private static int[] sortedArray(int size) {
        int[] arr = new int[size];
        for (int i = 0; i < size; i++) {
            arr[i] = i;
        }
        return arr;
    }

    private static int[] reverseSortedArray(int size) {
        int[] arr = new int[size];
        for (int i = 0; i < size; i++) {
            arr[i] = size - i;
        }
        return arr;
    }

    private static int[] manyDuplicatesArray(int size) {
        Random rng = new Random(SEED);
        int[] arr = new int[size];
        // Only 10 distinct values to create heavy duplication
        for (int i = 0; i < size; i++) {
            arr[i] = rng.nextInt(10);
        }
        return arr;
    }
}

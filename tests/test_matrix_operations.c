#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "matrix_generation.h"
#include "matrix_compression.h"
#include "matrix_multiplication.h"
#include "timing.h"

// Function to generate a random dense matrix
int** generate_random_matrix(int rows, int cols, float density) {
    int** matrix = allocateMatrix(rows, cols);
    initialiseMatrix(matrix, rows, cols, density);
    return matrix;
}

// Function to test parallel matrix multiplication
void test_parallel_matrix_multiplication(int rows_a, int cols_a, int cols_b, float density) {
    printf("\n==== Testing Parallel Matrix Multiplication ====\n");
    printf("Matrix A: %d x %d\n", rows_a, cols_a);
    printf("Matrix B: %d x %d\n", cols_a, cols_b);
    printf("Density: %.2f\n", density);

    // Generate and compress matrices
    int** dense_a = generate_random_matrix(rows_a, cols_a, density);
    int** dense_b = generate_random_matrix(cols_a, cols_b, density);
    CompressedMatrix* compressed_a = compress_matrix(dense_a, rows_a, cols_a, density);
    CompressedMatrix* compressed_b = compress_matrix(dense_b, cols_a, cols_b, density);

    // Get maximum number of threads
    int max_threads = omp_get_max_threads();
    printf("Maximum number of threads available: %d\n", max_threads);

    // Test with different numbers of threads
    for (int num_threads = 1; num_threads <= max_threads; num_threads *= 2) {
        omp_set_num_threads(num_threads);

        printf("\n--- Using %d thread(s) ---\n", num_threads);

        TICK(multiply_time);
        DenseMatrix* result = multiply_matrices(compressed_a, compressed_b);
        TOCK(multiply_time);

        // Verify result (for small matrices only)
        if (rows_a <= 10 && cols_b <= 10) {
            printf("Result matrix:\n");
            print_dense_matrix(result);
        }

        // Clean up result
        free_dense_matrix(result);
    }

    // Clean up
    freeMatrix(dense_a, rows_a);
    freeMatrix(dense_b, cols_a);
    free_compressed_matrix(compressed_a);
    free_compressed_matrix(compressed_b);
}

int main() {
    srand(time(NULL));  // Seed the random number generator

    printf("===== Parallel Matrix Multiplication Tests =====\n");

    // Test with small matrices
    test_parallel_matrix_multiplication(10, 10, 10, 0.5);

    // Test with medium-sized matrices
    test_parallel_matrix_multiplication(100, 100, 100, 0.1);

    // Test with large matrices
    test_parallel_matrix_multiplication(1000, 1000, 1000, 0.01);
    test_parallel_matrix_multiplication(10000, 10000, 10000, 0.01);

    // Test with project requirement matrices
    // Note: These may take a long time to run
    // test_parallel_matrix_multiplication(100000, 100000, 100000, 0.01);
    // test_parallel_matrix_multiplication(100000, 100000, 100000, 0.02);
    // test_parallel_matrix_multiplication(100000, 100000, 100000, 0.05);

    printf("\n===== All tests completed =====\n");

    return 0;
}
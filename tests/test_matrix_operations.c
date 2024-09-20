#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
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

// Function to test parallel matrix multiplication with logging
void test_parallel_matrix_multiplication(int rows_a, int cols_a, int cols_b, float density, FILE* log_file) {
    fprintf(log_file, "\n==== Testing Parallel Matrix Multiplication ====\n");
    fprintf(log_file, "Matrix A: %d x %d\n", rows_a, cols_a);
    fprintf(log_file, "Matrix B: %d x %d\n", cols_a, cols_b);
    fprintf(log_file, "Density: %.2f\n", density);

    // Generate and compress matrices
    int** dense_a = generate_random_matrix(rows_a, cols_a, density);
    int** dense_b = generate_random_matrix(cols_a, cols_b, density);
    CompressedMatrix* compressed_a = compress_matrix(dense_a, rows_a, cols_a, density);
    CompressedMatrix* compressed_b = compress_matrix(dense_b, cols_a, cols_b, density);

    // Get maximum number of threads
    int max_threads = omp_get_max_threads();
    fprintf(log_file, "Maximum number of threads available: %d\n", max_threads);

    // CSV header for easy data extraction
    fprintf(log_file, "Threads,CPU Time (s),Wall Clock Time (s)\n");

    // Test with different numbers of threads
    for (int num_threads = 1; num_threads <= max_threads; num_threads *= 2) {
        omp_set_num_threads(num_threads);

        fprintf(log_file, "\n--- Using %d thread(s) ---\n", num_threads);

        TICK(multiply_time);
        DenseMatrix* result = multiply_matrices(compressed_a, compressed_b);
        TOCK(multiply_time);

        // Log results in CSV format
        fprintf(log_file, "%d,%.6f,%.6f\n", num_threads, multiply_time.cpu_time, multiply_time.wall_time);

        // Verify result (for small matrices only)
        if (rows_a <= 10 && cols_b <= 10) {
            fprintf(log_file, "Result matrix:\n");
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

    // Get and print current working directory
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("Current working directory: %s\n", cwd);
    } else {
        perror("getcwd() error");
        return 1;
    }

    // Construct full path for log file
    char log_file_path[2048];  // Increased buffer size
    int path_length = snprintf(log_file_path, sizeof(log_file_path),
                               "%s/matrix_multiplication_results.log", cwd);
    if (path_length < 0 || path_length >= sizeof(log_file_path)) {
        fprintf(stderr, "Error: Log file path is too long.\n");
        return 1;
    }
    printf("Attempting to create log file at: %s\n", log_file_path);

    // Open log file
    FILE* log_file = fopen(log_file_path, "w");
    if (log_file == NULL) {
        fprintf(stderr, "Error opening log file: %s\n", strerror(errno));
        return 1;
    }

    fprintf(log_file, "===== Parallel Matrix Multiplication Tests =====\n");

    // Test with small matrices
    test_parallel_matrix_multiplication(10, 10, 10, 0.5, log_file);

    // Test with medium-sized matrices
    test_parallel_matrix_multiplication(100, 100, 100, 0.1, log_file);

    // Test with large matrices
    test_parallel_matrix_multiplication(1000, 1000, 1000, 0.01, log_file);
    test_parallel_matrix_multiplication(10000, 10000, 10000, 0.01, log_file);

    // Test with project requirement matrices
    // Note: These may take a long time to run
    // test_parallel_matrix_multiplication(100000, 100000, 100000, 0.01, log_file);
    // test_parallel_matrix_multiplication(100000, 100000, 100000, 0.02, log_file);
    // test_parallel_matrix_multiplication(100000, 100000, 100000, 0.05, log_file);

    fprintf(log_file, "\n===== All tests completed =====\n");

    // Close log file
    if (fclose(log_file) != 0) {
        printf("Error closing log file: %s\n", strerror(errno));
    }

    printf("Testing completed. Results written to %s\n", log_file_path);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include "matrix_generation.h"
#include "matrix_compression.h"
#include "matrix_multiplication.h"
#include "timing.h"

// Function to create directories
void create_directory(const char* path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        mkdir(path, 0700);
    }
}

// Function to generate a random dense matrix
int** generate_random_matrix(int rows, int cols, float density) {
    int** matrix = allocateMatrix(rows, cols);
    initialiseMatrix(matrix, rows, cols, density);
    return matrix;
}

CompressedMatrix* compress_matrix_and_write(int** matrix, size_t rows, size_t cols, float density, const char* dir_path) {
    CompressedMatrix* compressed = compress_matrix(matrix, rows, cols, density);

    char b_file_path[256];
    char c_file_path[256];
    snprintf(b_file_path, sizeof(b_file_path), "%s/B.txt", dir_path);
    snprintf(c_file_path, sizeof(c_file_path), "%s/C.txt", dir_path);

    FILE* b_file = fopen(b_file_path, "w");
    FILE* c_file = fopen(c_file_path, "w");

    if (b_file == NULL || c_file == NULL) {
        fprintf(stderr, "Error opening files for writing compressed matrices\n");
        exit(1);
    }

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < compressed->row_sizes[i]; j++) {
            fprintf(b_file, "%d ", compressed->B[i][j]);
            fprintf(c_file, "%d ", compressed->C[i][j]);
        }
        fprintf(b_file, "\n");
        fprintf(c_file, "\n");
    }

    fclose(b_file);
    fclose(c_file);

    return compressed;
}

// Function to test parallel matrix multiplication with logging
void test_parallel_matrix_multiplication(int rows_a, int cols_a, int cols_b, float density, const char* base_dir) {
    char log_dir[256];
    snprintf(log_dir, sizeof(log_dir), "%s/matrix_multiplication_%dx%dx%d_%.2f", base_dir, rows_a, cols_a, cols_b, density);

    // Try to create directory and check if successful
    if (mkdir(log_dir, 0700) != 0 && errno != EEXIST) {
        fprintf(stderr, "Error creating directory %s: %s\n", log_dir, strerror(errno));
        return;
    }

    char matrix_a_dir[256], matrix_b_dir[256];
    snprintf(matrix_a_dir, sizeof(matrix_a_dir), "%s/matrix_a", log_dir);
    snprintf(matrix_b_dir, sizeof(matrix_b_dir), "%s/matrix_b", log_dir);

    // Try to create directories and check if successful
    if ((mkdir(matrix_a_dir, 0700) != 0 && errno != EEXIST) ||
        (mkdir(matrix_b_dir, 0700) != 0 && errno != EEXIST)) {
        fprintf(stderr, "Error creating matrix directories: %s\n", strerror(errno));
        return;
    }

    char performance_file[256];
    snprintf(performance_file, sizeof(performance_file), "%s/performance.txt", log_dir);

    // Try to open performance file
    FILE* perf_file = fopen(performance_file, "w");
    if (perf_file == NULL) {
        fprintf(stderr, "Error opening performance file %s: %s\n", performance_file, strerror(errno));
        perror("fopen");
        return;
    }

    // Print current working directory and full path of performance file
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        fprintf(stderr, "Current working directory: %s\n", cwd);
    } else {
        perror("getcwd");
    }
    fprintf(stderr, "Attempting to write to file: %s\n", performance_file);

    fprintf(perf_file, "Matrix A: %d x %d\n", rows_a, cols_a);
    fprintf(perf_file, "Matrix B: %d x %d\n", cols_a, cols_b);
    fprintf(perf_file, "Density: %.2f\n\n", density);

    // Generate and compress matrices
    int** dense_a = generate_random_matrix(rows_a, cols_a, density);
    CompressedMatrix* compressed_a = compress_matrix_and_write(dense_a, rows_a, cols_a, density, matrix_a_dir);
    freeMatrix(dense_a, rows_a);

    int** dense_b = generate_random_matrix(cols_a, cols_b, density);
    CompressedMatrix* compressed_b = compress_matrix_and_write(dense_b, cols_a, cols_b, density, matrix_b_dir);
    freeMatrix(dense_b, cols_a);

    // Get maximum number of threads
    int max_threads = omp_get_max_threads();
    fprintf(perf_file, "Maximum number of threads available: %d\n\n", max_threads);

    // CSV header for easy data extraction
    fprintf(perf_file, "Threads,CPU Time (s),Wall Clock Time (s)\n");

    // Test with different numbers of threads
    for (int num_threads = 1; num_threads <= max_threads; num_threads *= 2) {
        omp_set_num_threads(num_threads);

        TICK(multiply_time);
        DenseMatrix* result = multiply_matrices(compressed_a, compressed_b);
        TOCK(multiply_time);

        // Log results in CSV format
        fprintf(perf_file, "%d,%.6f,%.6f\n", num_threads, multiply_time.cpu_time, multiply_time.wall_time);

        // Clean up result
        free_dense_matrix(result);
    }

    // Clean up
    free_compressed_matrix(compressed_a);
    free_compressed_matrix(compressed_b);
    fclose(perf_file);
}

int main() {
    srand(time(NULL));  // Seed the random number generator

    char base_dir[256];
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    snprintf(base_dir, sizeof(base_dir), "proj/logging/matrix_multiplication_%02d%02d%d",
             tm->tm_mday, tm->tm_mon + 1, tm->tm_year + 1900);
    create_directory("proj");
    create_directory("proj/logging");
    create_directory(base_dir);

    // Print the full path of the base directory
    char full_path[1024];
    if (realpath(base_dir, full_path) != NULL) {
        printf("Full path of base directory: %s\n", full_path);
    } else {
        perror("realpath");
    }

    test_parallel_matrix_multiplication(1000, 1000, 1000, 0.01, base_dir);

    // Test with project requirement matrices
    // test_parallel_matrix_multiplication(10000, 100000, 100000, 0.01, base_dir);
    // test_parallel_matrix_multiplication(100000, 100000, 100000, 0.02, base_dir);
    // test_parallel_matrix_multiplication(100000, 100000, 100000, 0.05, base_dir);

    printf("Testing completed. Results written to %s\n", base_dir);

    return 0;
}
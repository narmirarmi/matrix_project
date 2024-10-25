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


// Function to create directories with logging
int create_directory(const char* path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        if (mkdir(path, 0700) == -1) {
            fprintf(stderr, "Error creating directory %s: %s\n", path, strerror(errno));
            return -1;
        }
        printf("Created new directory: %s\n", path);
    } else {
        printf("Using existing directory: %s\n", path);
    }
    return 0;
}

// Function to generate a unique directory name
char* generate_unique_directory(const char* base_path) {
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    char *unique_dir = malloc(512 * sizeof(char));
    if (unique_dir == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }
    snprintf(unique_dir, 512, "%s_%04d%02d%02d_%02d%02d%02d",
             base_path, tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
             tm->tm_hour, tm->tm_min, tm->tm_sec);
    return unique_dir;
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
    char log_dir[512];
    snprintf(log_dir, sizeof(log_dir), "%s/matrix_multiplication_%dx%dx%d_%.2f", base_dir, rows_a, cols_a, cols_b, density);

    if (create_directory(log_dir) != 0) {
        fprintf(stderr, "Error creating log directory %s\n", log_dir);
        return;
    }

    char matrix_a_dir[512], matrix_b_dir[512];
    snprintf(matrix_a_dir, sizeof(matrix_a_dir), "%s/matrix_a", log_dir);
    snprintf(matrix_b_dir, sizeof(matrix_b_dir), "%s/matrix_b", log_dir);

    if (create_directory(matrix_a_dir) != 0 || create_directory(matrix_b_dir) != 0) {
        fprintf(stderr, "Error creating matrix directories\n");
        return;
    }

    printf("Generating and compressing matrices for density %.2f...\n", density);

    // Generate and compress matrices
    int** dense_a = generate_random_matrix(rows_a, cols_a, density);
    CompressedMatrix* compressed_a = compress_matrix_and_write(dense_a, rows_a, cols_a, density, matrix_a_dir);
    freeMatrix(dense_a, rows_a);

    int** dense_b = generate_random_matrix(cols_a, cols_b, density);
    CompressedMatrix* compressed_b = compress_matrix_and_write(dense_b, cols_a, cols_b, density, matrix_b_dir);
    freeMatrix(dense_b, cols_a);

    const char* schedule_names[] = {"static", "dynamic", "guided", "auto"};
    ScheduleType schedule_types[] = {SCHEDULE_STATIC, SCHEDULE_DYNAMIC, SCHEDULE_GUIDED, SCHEDULE_AUTO};
    int num_schedule_types = sizeof(schedule_types) / sizeof(schedule_types[0]);

    // Get maximum number of threads
    int max_threads = omp_get_max_threads();
    printf("Maximum number of threads available: %d\n", max_threads);

    // Test with different numbers of threads and scheduling types
    for (int num_threads = max_threads; num_threads >= 1; num_threads /= 2) {
        for (int s = 0; s < num_schedule_types; s++) {
            char performance_file[512];
            snprintf(performance_file, sizeof(performance_file), "%s/performance_%dx%dx%d_%.2f_threads_%d_%s.csv",
                     log_dir, rows_a, cols_a, cols_b, density, num_threads, schedule_names[s]);

            FILE* perf_file = fopen(performance_file, "w");
            if (perf_file == NULL) {
                fprintf(stderr, "Error opening performance file %s: %s\n", performance_file, strerror(errno));
                continue;
            }

            fprintf(perf_file, "Matrix A: %d x %d\n", rows_a, cols_a);
            fprintf(perf_file, "Matrix B: %d x %d\n", cols_a, cols_b);
            fprintf(perf_file, "Density: %.2f\n", density);
            fprintf(perf_file, "Threads: %d\n", num_threads);
            fprintf(perf_file, "Schedule: %s\n\n", schedule_names[s]);

            fprintf(perf_file, "CPU Time (s),Wall Clock Time (s)\n");

            // Explicitly set the number of threads for OpenMP
            omp_set_num_threads(num_threads);

            // Ensure the change takes effect
            #pragma omp parallel
            {
                #pragma omp single
                {
                    int actual_threads = omp_get_num_threads();
                    printf("Actually using %d thread(s) for density %.2f with %s scheduling...\n",
                           actual_threads, density, schedule_names[s]);
                }
            }

            TICK(multiply_time);
            DenseMatrix* result = multiply_matrices(compressed_a, compressed_b, schedule_types[s]);
            TOCK(multiply_time);

            // Log results in CSV format
            fprintf(perf_file, "%.6f,%.6f\n", multiply_time.cpu_time, multiply_time.wall_time);

            // Clean up result
            free_dense_matrix(result);

            fclose(perf_file);
            printf("Performance data for %d threads with %s scheduling written to %s\n",
                   num_threads, schedule_names[s], performance_file);
        }
    }

    // Clean up
    free_compressed_matrix(compressed_a);
    free_compressed_matrix(compressed_b);

    printf("Test completed for matrix size %dx%dx%d with density %.2f\n", rows_a, cols_a, cols_b, density);

    // Reset OpenMP to use the maximum number of threads
    omp_set_num_threads(max_threads);
}

int main() {

    srand(time(NULL));  // Seed the random number generator

    printf("Starting matrix multiplication test...\n");

    // Get the path to the project root directory (parent of build)
    char project_root[1024];
    if (getcwd(project_root, sizeof(project_root)) == NULL) {
        perror("getcwd");
        return 1;
    }
    char *build_pos = strstr(project_root, "/build");
    if (build_pos != NULL) {
        *build_pos = '\0';  // Truncate at "/build"
    }

    // Create logs directory in the project root
    char logs_dir[1024];
    snprintf(logs_dir, sizeof(logs_dir), "%s/logs", project_root);
    if (create_directory(logs_dir) != 0) {
        fprintf(stderr, "Failed to create logs directory\n");
        return 1;
    }

    // Generate unique directory name for this run
    char *run_dir_name = generate_unique_directory("run");
    if (run_dir_name == NULL) {
        fprintf(stderr, "Failed to generate unique directory name\n");
        return 1;
    }

    // Create the full path for the run directory
    char run_dir_path[1024];
    snprintf(run_dir_path, sizeof(run_dir_path), "%s/%s", logs_dir, run_dir_name);
    if (create_directory(run_dir_path) != 0) {
        fprintf(stderr, "Failed to create run directory\n");
        free(run_dir_name);
        return 1;
    }

    // Print the full path of the run directory
    printf("Full path of test directory: %s\n", run_dir_path);

    printf("Beginning matrix multiplication tests...\n");

    // Test with smaller matrices
    test_parallel_matrix_multiplication(100000, 10000, 10000, 0.01f, run_dir_path);
    test_parallel_matrix_multiplication(100000, 10000, 10000, 0.02f, run_dir_path);

    // test_parallel_matrix_multiplication(100000, 100000, 100000, 0.01f, run_dir_path);
    // test_parallel_matrix_multiplication(100000, 100000, 100000, 0.02f, run_dir_path);
    // test_parallel_matrix_multiplication(100000, 100000, 100000, 0.05f, run_dir_path);

    printf("All tests completed. Results written to %s\n", run_dir_path);

    free(run_dir_name);
    return 0;
}
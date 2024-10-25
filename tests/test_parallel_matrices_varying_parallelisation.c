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

#define MAX_TIME_SECONDS 650
#define NUM_RUNS 1
#define DEFAULT_DENSITY 0.05
#define DEFAULT_SIZE 40000

// Function to get parallelisation type name
const char* get_parallelisation_name(parallelisation_type type) {
    switch(type) {
        case MULT_SEQUENTIAL: return "sequential";
        case MULT_OMP: return "openmp";
        case MULT_MPI: return "mpi";
        default: return "unknown";
    }
}

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

void test_parallel_matrix_multiplication(int rows_a, int cols_a, int cols_b, float density,
                                      const char* base_dir, parallelisation_type parallel_type) {
    // Get parallelisation name for logging
    const char* parallel_name = get_parallelisation_name(parallel_type);

    char log_dir[512];
    snprintf(log_dir, sizeof(log_dir), "%s/matrix_multiplication_%dx%dx%d_%.2f_%s",
             base_dir, rows_a, cols_a, cols_b, density, parallel_name);

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

    printf("Generating and compressing matrices for density %.2f using %s...\n", density, parallel_name);

    // Generate and compress matrices
    int** dense_a = generate_random_matrix(rows_a, cols_a, density);
    CompressedMatrix* compressed_a = compress_matrix_and_write(dense_a, rows_a, cols_a, density, matrix_a_dir);
    freeMatrix(dense_a, rows_a);

    int** dense_b = generate_random_matrix(cols_a, cols_b, density);
    CompressedMatrix* compressed_b = compress_matrix_and_write(dense_b, cols_a, cols_b, density, matrix_b_dir);
    freeMatrix(dense_b, cols_a);

    // Get maximum number of threads for OpenMP
    const int max_threads = omp_get_max_threads();
    if (parallel_type != MULT_SEQUENTIAL) {
        printf("Maximum number of threads available: %d\n", max_threads);
    }
    char performance_file[512];
    snprintf(performance_file, sizeof(performance_file), "%s/performance_%dx%dx%d_%.2f_%s.csv",
             log_dir, rows_a, cols_a, cols_b, density, parallel_name);

    FILE* perf_file = fopen(performance_file, "w");
    if (perf_file == NULL) {
        fprintf(stderr, "Error opening performance file %s: %s\n", performance_file, strerror(errno));
        return;
    }

    // Write header information
    fprintf(perf_file, "Matrix A: %d x %d\n", rows_a, cols_a);
    fprintf(perf_file, "Matrix B: %d x %d\n", cols_a, cols_b);
    fprintf(perf_file, "Density: %.2f\n", density);
    fprintf(perf_file, "Parallelisation: %s\n\n", parallel_name);
    fprintf(perf_file, "CPU Time (s),Wall Clock Time (s)\n");

    // Set thread count for OpenMP
    if (parallel_type == MULT_OMP) {
        omp_set_num_threads(max_threads);
        #pragma omp parallel
        {
            #pragma omp single
            {
                printf("Using %d thread(s) for density %.2f with %s...\n",
                       omp_get_num_threads(), density, parallel_name);
            }
        }
    }

    // Perform multiplication and timing
    TICK(multiply_time);
    DenseMatrix* result = multiply_matrices(compressed_a, compressed_b, parallel_type);
    TOCK(multiply_time);

    // Log timing results
    fprintf(perf_file, "%.6f,%.6f\n", multiply_time.cpu_time, multiply_time.wall_time);

    // Clean up
    free_dense_matrix(result);
    fclose(perf_file);
    free_compressed_matrix(compressed_a);
    free_compressed_matrix(compressed_b);

    printf("Test completed for matrix size %dx%dx%d with density %.2f using %s\n",
           rows_a, cols_a, cols_b, density, parallel_name);
    printf("Performance data written to %s\n", performance_file);
}

int main(int argc, char** argv) {
    srand(time(NULL));

    int opt;
    parallelisation_type parallel_type = MULT_SEQUENTIAL;
    int gen_size = DEFAULT_SIZE;
    float density = DEFAULT_DENSITY;

    while((opt = getopt(argc, argv, ":s:omt:")) != -1) {
        switch(opt) {
            case 's':
                gen_size = atoi(optarg);
                break;
            case 'o':
                parallel_type = MULT_OMP;
                break;
            case 'm':
                parallel_type = MULT_MPI;
                break;
            case '?':
                printf("FLAGS:\n\t-s [size]: set matrix size\n\t-o: use OpenMP\n\t-m: use MPI\n");
                return 1;
        }
    }

    printf("Profiling matrix multiplication using %s\n", get_parallelisation_name(parallel_type));
    printf("SIZE: %d\tDENSITY: %.2f\n", gen_size, density);

    // Set up directories
    char project_root[1024];
    if (getcwd(project_root, sizeof(project_root)) == NULL) {
        perror("getcwd");
        return 1;
    }

    char *build_pos = strstr(project_root, "/build");
    if (build_pos != NULL) {
        *build_pos = '\0';
    }

    char logs_dir[1024];
    snprintf(logs_dir, sizeof(logs_dir), "%s/logs", project_root);
    if (create_directory(logs_dir) != 0) {
        fprintf(stderr, "Failed to create logs directory\n");
        return 1;
    }

    char *run_dir_name = generate_unique_directory("run");
    if (run_dir_name == NULL) {
        fprintf(stderr, "Failed to generate unique directory name\n");
        return 1;
    }

    char run_dir_path[1024];
    snprintf(run_dir_path, sizeof(run_dir_path), "%s/%s", logs_dir, run_dir_name);
    if (create_directory(run_dir_path) != 0) {
        fprintf(stderr, "Failed to create run directory\n");
        free(run_dir_name);
        return 1;
    }

    printf("Test directory: %s\n", run_dir_path);

    // Run the test with specified parameters
    test_parallel_matrix_multiplication(gen_size, gen_size, gen_size, density, run_dir_path, parallel_type);

    printf("Test completed. Results written to %s\n", run_dir_path);
    free(run_dir_name);
    return 0;
}
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
#include "mpi_matrix_compression.h"
#include "matrix_multiplication.h"
#include "timing.h"
#include <mpi.h>
#include <limits.h>
#include <libgen.h>

#define MAX_TIME_SECONDS 650
#define NUM_RUNS 1
#define DEFAULT_DENSITY 0.02
#define DEFAULT_SIZE 40000

char* get_project_root() {
    char* project_root = malloc(PATH_MAX * sizeof(char));
    if (project_root == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    // Get the path to the current executable
    char exec_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exec_path, sizeof(exec_path) - 1);
    if (len == -1) {
        fprintf(stderr, "Error getting executable path: %s\n", strerror(errno));
        free(project_root);
        return NULL;
    }
    exec_path[len] = '\0';

    // Get the directory containing the executable
    char* dir = dirname(exec_path);

    // Navigate up to the project root (assuming the executable is in a build directory)
    snprintf(project_root, PATH_MAX, "%s/../..", dir);

    // Resolve the path to remove ../ components
    char resolved_path[PATH_MAX];
    if (realpath(project_root, resolved_path) == NULL) {
        fprintf(stderr, "Error resolving path: %s\n", strerror(errno));
        free(project_root);
        return NULL;
    }

    strcpy(project_root, resolved_path);
    return project_root;
}

void debug_print_paths(const char* label, const char* path) {
    char resolved_path[PATH_MAX];
    if (realpath(path, resolved_path) != NULL) {
        printf("[DEBUG] %s (resolved): %s\n", label, resolved_path);
    } else {
        printf("[DEBUG] %s (raw): %s\n", label, path);
        printf("[DEBUG] %s (error): %s\n", label, strerror(errno));
    }
}

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

void write_compressed_matrix(CompressedMatrix* compressed, size_t rows, const char* dir_path) {
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
}

void test_parallel_matrix_multiplication(int rows_a, int cols_a, int cols_b, float density,
                                      const char* base_dir, parallelisation_type parallel_type) {
    int rank = 0, size = 1;
    if (parallel_type == MULT_MPI) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    // Seed the random number generator
    srand(42 + rank);  // Different seed for each process

    // Get parallelisation name for logging
    const char* parallel_name = get_parallelisation_name(parallel_type);

    // Define directory paths
    char log_dir[512] = "";
    char matrix_a_dir[512] = "";
    char matrix_b_dir[512] = "";

    // Only the root process performs directory creation and file I/O
    if (rank == 0) {
        fprintf(stderr, "Root process here");

        // Generate the log directory path
        snprintf(log_dir, sizeof(log_dir), "%s/matrix_multiplication_%dx%dx%d_%.2f_%s",
                 base_dir, rows_a, cols_a, cols_b, density, parallel_name);

        // Create the log directory
        if (create_directory(log_dir) != 0) {
            fprintf(stderr, "Error creating log directory %s\n", log_dir);
            return;
        }

        // Create directories for matrices A and B
        snprintf(matrix_a_dir, sizeof(matrix_a_dir), "%s/matrix_a", log_dir);
        snprintf(matrix_b_dir, sizeof(matrix_b_dir), "%s/matrix_b", log_dir);

        if (create_directory(matrix_a_dir) != 0 || create_directory(matrix_b_dir) != 0) {
            fprintf(stderr, "Error creating matrix directories\n");
            return;
        }
    }

    // Broadcast the directories to all processes
    if (parallel_type == MULT_MPI) {
        MPI_Bcast(log_dir, 512, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(matrix_a_dir, 512, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(matrix_b_dir, 512, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    printf("Process %d: Generating and compressing matrices for density %.2f using %s...\n", rank, density, parallel_name);

    // Generate and compress matrices on all processes
    int** dense_a = NULL;
    int** dense_b = NULL;
    CompressedMatrix* compressed_a = NULL;
    CompressedMatrix* compressed_b = NULL;

    if (parallel_type == MULT_MPI) {
        // Calculate local rows for matrix A
        size_t local_rows_a = rows_a / size;
        size_t remainder_a = rows_a % size;
        if ((size_t)rank < remainder_a) {
            local_rows_a++;
        }

        // Calculate local rows for matrix B (columns of A)
        size_t local_rows_b = cols_a / size;
        size_t remainder_b = cols_a % size;
        if ((size_t)rank < remainder_b) {
            local_rows_b++;
        }

        // Generate local matrices
        dense_a = allocateMatrix(local_rows_a, cols_a);
        initialiseMatrix(dense_a, local_rows_a, cols_a, density);

        dense_b = allocateMatrix(local_rows_b, cols_b);
        initialiseMatrix(dense_b, local_rows_b, cols_b, density);

        // Compress matrices using MPI functions
        compressed_a = compress_matrix_with_mpi(dense_a, local_rows_a, cols_a, density, MPI_COMM_WORLD);
        compressed_b = compress_matrix_with_mpi(dense_b, local_rows_b, cols_b, density, MPI_COMM_WORLD);

        // Free local dense matrices
        freeMatrix(dense_a, local_rows_a);
        freeMatrix(dense_b, local_rows_b);

    } else {
        // Non-MPI version
        dense_a = generate_random_matrix(rows_a, cols_a, density);
        compressed_a = compress_matrix(dense_a, rows_a, cols_a, density);

        dense_b = generate_random_matrix(cols_a, cols_b, density);
        compressed_b = compress_matrix(dense_b, cols_a, cols_b, density);

        // Free the dense matrices after compression
        freeMatrix(dense_a, rows_a);
        freeMatrix(dense_b, cols_a);
    }

    // Only the root process writes the compressed matrices to files
    if (rank == 0) {
        // Write the compressed matrices to files
        write_compressed_matrix(compressed_a, compressed_a->num_rows, matrix_a_dir);
        write_compressed_matrix(compressed_b, compressed_b->num_rows, matrix_b_dir);
    }

    // Get maximum number of threads for OpenMP
    const int max_threads = omp_get_max_threads();
    if (parallel_type == MULT_OMP) {
        printf("Process %d: Maximum number of threads available: %d\n", rank, max_threads);
    }

    // Performance file variables
    FILE* perf_file = NULL;
    char performance_file[512] = "";

    // Only the root process handles performance logging
    if (rank == 0) {
        snprintf(performance_file, sizeof(performance_file), "%s/performance_%dx%dx%d_%.2f_%s.csv",
                 log_dir, rows_a, cols_a, cols_b, density, parallel_name);

        perf_file = fopen(performance_file, "w");
        if (perf_file == NULL) {
            fprintf(stderr, "Error opening performance file %s: %s\n", performance_file, strerror(errno));
            return;
        }

        // Write header information to the performance file
        fprintf(perf_file, "Matrix A: %d x %d\n", rows_a, cols_a);
        fprintf(perf_file, "Matrix B: %d x %d\n", cols_a, cols_b);
        fprintf(perf_file, "Density: %.2f\n", density);
        fprintf(perf_file, "Parallelisation: %s\n\n", parallel_name);
        fprintf(perf_file, "CPU Time (s),Wall Clock Time (s)\n");
    }

    // Set thread count for OpenMP
    if (parallel_type == MULT_OMP) {
        omp_set_num_threads(10);
        #pragma omp parallel
        {
            #pragma omp single
            {
                printf("Process %d: Using %d thread(s) for density %.2f with %s...\n",
                       rank, omp_get_num_threads(), density, parallel_name);
            }
        }
    }

    // Perform multiplication and timing
    TICK(multiply_time);
    DenseMatrix* result = NULL;

    if (parallel_type == MULT_MPI) {
        result = multiply_matrices(compressed_a, compressed_b, parallel_type);
    } else {
        result = multiply_matrices(compressed_a, compressed_b, parallel_type);
    }
    TOCK(multiply_time);

    // Only the root process logs timing results
    if (rank == 0 && perf_file != NULL) {
        fprintf(perf_file, "%.6f,%.6f\n", multiply_time.cpu_time, multiply_time.wall_time);
        fclose(perf_file);
    }

    // Clean up
    if (result != NULL) {
        if (parallel_type == MULT_MPI) {
            free_dense_matrix(result); // Adjust if needed for MPI
        } else {
            free_dense_matrix(result);
        }
    }
    free_compressed_matrix(compressed_a);
    free_compressed_matrix(compressed_b);

    // Only the root process prints the final messages
    if (rank == 0) {
        printf("Test completed for matrix size %dx%dx%d with density %.2f using %s\n",
               rows_a, cols_a, cols_b, density, parallel_name);
        printf("Performance data written to %s\n", performance_file);
    }
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

    int rank = 0, size = 1;
    if (parallel_type == MULT_MPI) {
        // Initialize MPI
        MPI_Init(&argc, &argv);
        // Get the rank and size
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        printf("Process %d of %d initialized\n", rank, size);
    }

    // Only the root process should perform directory setup
    char run_dir_path[PATH_MAX] = "";
    if (rank == 0) {
        printf("Profiling matrix multiplication using %s\n", get_parallelisation_name(parallel_type));
        printf("SIZE: %d\tDENSITY: %.2f\n", gen_size, density);

        // Get project root directory
        char* project_root = get_project_root();
        if (project_root == NULL) {
            fprintf(stderr, "Failed to determine project root directory\n");
            return 1;
        }

        // Create logs directory in project root
        char logs_dir[PATH_MAX];
        snprintf(logs_dir, sizeof(logs_dir), "%s/logs", project_root);
        if (create_directory(logs_dir) != 0) {
            fprintf(stderr, "Failed to create logs directory\n");
            free(project_root);
            return 1;
        }

        char* run_dir_name = generate_unique_directory("run");
        if (run_dir_name == NULL) {
            fprintf(stderr, "Failed to generate unique directory name\n");
            free(project_root);
            return 1;
        }

        snprintf(run_dir_path, sizeof(run_dir_path), "%s/%s", logs_dir, run_dir_name);
        if (create_directory(run_dir_path) != 0) {
            fprintf(stderr, "Failed to create run directory\n");
            free(project_root);
            free(run_dir_name);
            return 1;
        }

        printf("Test directory: %s\n", run_dir_path);
        free(project_root);
        free(run_dir_name);
    }

    // Broadcast the run directory path to all processes
    if (parallel_type == MULT_MPI) {
        MPI_Bcast(run_dir_path, PATH_MAX, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    // Run the test with specified parameters
    test_parallel_matrix_multiplication(gen_size, gen_size, gen_size, density, run_dir_path, parallel_type);

    // Finalize MPI
    if (parallel_type == MULT_MPI) {
        MPI_Finalize();
    }

    if (rank == 0) {
        printf("Test completed. Results written to %s\n", run_dir_path);
    }

    return 0;
}
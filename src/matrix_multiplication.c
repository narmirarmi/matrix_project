#include <mpi.h>
#include "matrix_multiplication.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "timing.h"
#include <omp.h>

DenseMatrix* multiply_matrices(const CompressedMatrix* A, const CompressedMatrix* B, parallelisation_type parallelisation_type) {
    if (A->num_cols != B->num_rows) {
        fprintf(stderr, "Error: Incompatible matrix dimensions for multiplication\n");
        return NULL;
    }

    DenseMatrix* result = malloc(sizeof(DenseMatrix));
    result->rows = A->num_rows;
    result->cols = B->num_cols;
    result->data = malloc(result->rows * sizeof(int*));

    for (size_t i = 0; i < result->rows; i++) {
        result->data[i] = calloc(result->cols, sizeof(int));
    }

    // Start timing
    TICK(multiply_time);

    // Perform matrix multiplication with Sequential, OMP or MPI multiplication
    switch (parallelisation_type) {
        case MULT_SEQUENTIAL:for (size_t i = 0; i < A->num_rows; i++) {
                for (size_t k = 0; k < A->row_sizes[i]; k++) {
                    int a_val = A->B[i][k];
                    size_t a_col = A->C[i][k];
                    for (size_t j = 0; j < B->row_sizes[a_col]; j++) {
                        size_t b_col = B->C[a_col][j];
                        int b_val = B->B[a_col][j];
                        result->data[i][b_col] += a_val * b_val;
                    }
                }
            }
            break;

        case MULT_OMP:
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < A->num_rows; i++) {
                for (size_t k = 0; k < A->row_sizes[i]; k++) {
                    int a_val = A->B[i][k];
                    size_t a_col = A->C[i][k];
                    for (size_t j = 0; j < B->row_sizes[a_col]; j++) {
                        size_t b_col = B->C[a_col][j];
                        int b_val = B->B[a_col][j];
                        #pragma omp atomic
                        result->data[i][b_col] += a_val * b_val;
                    }
                }
            }
            break;

        case MULT_MPI: {
            int rank, size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            // Debug output for initial state
            printf("[Process %d] Starting MPI multiplication\n", rank);
            printf("[Process %d] Matrix A: %zu x %zu, Matrix B: %zu x %zu\n",
                   rank, A->num_rows, A->num_cols, B->num_rows, B->num_cols);

            // Validate input matrices
            if (A == NULL || B == NULL) {
                fprintf(stderr, "[Process %d] Error: NULL matrix pointer\n", rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return NULL;
            }

            // Validate matrix structures
            if (A->B == NULL || A->C == NULL || B->B == NULL || B->C == NULL) {
                fprintf(stderr, "[Process %d] Error: NULL matrix data pointers\n", rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return NULL;
            }

            // Debug: Print row sizes for first few rows
            printf("[Process %d] First row sizes - A: %zu, B: %zu\n",
                   rank, A->row_sizes[0], B->row_sizes[0]);

            // Calculate work distribution
            size_t rows_per_proc = A->num_rows / size;
            size_t remainder = A->num_rows % size;
            size_t start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);
            size_t num_rows = rows_per_proc + (rank < remainder ? 1 : 0);
            size_t end_row = start_row + num_rows;

            printf("[Process %d] Work distribution: rows %zu to %zu (total: %zu)\n",
                    rank, start_row, end_row - 1, num_rows);

            // Allocate temporary buffer for local results
            printf("[Process %d] Allocating local result buffer for %zu rows\n", rank, num_rows);
            int** local_result = malloc(num_rows * sizeof(int*));
            if (local_result == NULL) {
                fprintf(stderr, "[Process %d] Failed to allocate local result rows\n", rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return NULL;
            }

            for (size_t i = 0; i < num_rows; i++) {
                local_result[i] = calloc(result->cols, sizeof(int));
                if (local_result[i] == NULL) {
                    fprintf(stderr, "[Process %d] Failed to allocate local result columns\n", rank);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return NULL;
                }
            }

            // Perform local computation
            printf("[Process %d] Starting local computation\n", rank);
            for (size_t i = start_row; i < end_row; i++) {
                size_t local_i = i - start_row;

                // Debug: Periodically print progress
                if (i % 1000 == 0) {
                    printf("[Process %d] Processing row %zu\n", rank, i);
                }

                if (i >= A->num_rows) {
                    fprintf(stderr, "[Process %d] Error: Row index %zu exceeds matrix dimensions\n",
                            rank, i);
                    continue;
                }

                for (size_t k = 0; k < A->row_sizes[i]; k++) {
                    int a_val = A->B[i][k];
                    size_t a_col = A->C[i][k];

                    if (a_col >= B->num_rows) {
                        fprintf(stderr, "[Process %d] Error: Invalid column index %zu in matrix A\n",
                                rank, a_col);
                        continue;
                    }

                    for (size_t j = 0; j < B->row_sizes[a_col]; j++) {
                        size_t b_col = B->C[a_col][j];

                        if (b_col >= result->cols) {
                            fprintf(stderr, "[Process %d] Error: Invalid column index %zu in matrix B\n",
                                    rank, b_col);
                            continue;
                        }

                        int b_val = B->B[a_col][j];
                        result->data[i][b_col] += a_val * b_val;
                    }
                }
            }

            printf("[Process %d] Local computation completed\n", rank);

            // Gather results
            printf("[Process %d] Starting result gathering\n", rank);
            for (size_t i = 0; i < num_rows; i++) {
                size_t global_row = start_row + i;
                printf("[Process %d] Reducing row %zu\n", rank, global_row);

                MPI_Reduce(local_result[i], result->data[global_row],
                          result->cols, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            }

            printf("[Process %d] Result gathering completed\n", rank);

            // Clean up local memory
            printf("[Process %d] Cleaning up local memory\n", rank);
            for (size_t i = 0; i < num_rows; i++) {
                free(local_result[i]);
            }
            free(local_result);

            // Synchronize all processes
            MPI_Barrier(MPI_COMM_WORLD);
            printf("[Process %d] MPI multiplication completed\n", rank);
            break;
        }
    }

    TOCK(multiply_time);
    return result;
}

void free_dense_matrix(DenseMatrix* matrix) {
    for (size_t i = 0; i < matrix->rows; i++) {
        free(matrix->data[i]);
    }
    free(matrix->data);
    free(matrix);
}

void print_dense_matrix(const DenseMatrix* matrix) {
    for (size_t i = 0; i < matrix->rows; i++) {
        for (size_t j = 0; j < matrix->cols; j++) {
            printf("%2d ", matrix->data[i][j]);
        }
        printf("\n");
    }
}
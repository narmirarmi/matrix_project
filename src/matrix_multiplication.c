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

        case MULT_MPI:
            multiply_matrices_mpi(A, B, result);
            break;
    }

    TOCK(multiply_time);
    return result;
}

void multiply_matrices_mpi(const CompressedMatrix* A, const CompressedMatrix* B, DenseMatrix* result) {

    // mpi stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("computing from rank %d out of %d processors\n",
        rank, size);

    //get rows per process and the remainder
    size_t rows_per_proc = A->num_rows / size;
    size_t remainder = A->num_rows % size;

    // Calculate start and end rows for this process
    size_t start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    size_t num_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    size_t end_row = start_row + num_rows;

    // Allocate local result matrix
    int** local_result = (int**)malloc(num_rows * sizeof(int*));
    for (size_t i = 0; i < num_rows; i++) {
        local_result[i] = (int*)calloc(result->cols, sizeof(int));
    }

    //perform local computation
    for (size_t i = start_row; i < end_row; i++) {
        size_t local_i = i - start_row;  // Convert to local index
        for (size_t k = 0; k < A->row_sizes[i]; k++) {
            int a_val = A->B[i][k];
            size_t a_col = A->C[i][k];
            for (size_t j = 0; j < B->row_sizes[a_col]; j++) {
                size_t b_col = B->C[a_col][j];
                int b_val = B->B[a_col][j];
                local_result[local_i][b_col] += a_val * b_val;
            }
        }
    }

    //get results from all processes
    for (size_t i = 0; i < num_rows; i++) {
        size_t global_row = start_row + i;
        MPI_Allreduce(MPI_IN_PLACE, local_result[i], result->cols,
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Copy local result to global result matrix
        if (rank == 0) {
            memcpy(result->data[global_row], local_result[i],
                   result->cols * sizeof(int));
        }
    }

    // Broadcast the complete result to all processes
    for (size_t i = 0; i < A->num_rows; i++) {
        MPI_Bcast(result->data[i], result->cols, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Clean up local memory
    for (size_t i = 0; i < num_rows; i++) {
        free(local_result[i]);
    }

    free(local_result);
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
#include "matrix_multiplication.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "timing.h"
#include <omp.h>
#include <mpi.h>

DenseMatrix* multiply_matrices(const CompressedMatrix* A, const CompressedMatrix* B, parallelisation_type parallelisation_type) {
    if (A->num_cols != B->num_rows) {
        fprintf(stderr, "Error: Incompatible matrix dimensions for multiplication\n");
        return NULL;
    }

    int rank = 0, size = 1;
    if (parallelisation_type == MULT_MPI) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    DenseMatrix* result = malloc(sizeof(DenseMatrix));
    result->rows = A->num_rows;
    result->cols = B->num_cols;

    if (parallelisation_type != MULT_MPI || rank == 0) {
        // Allocate the full result matrix on the root process
        result->data = malloc(result->rows * sizeof(int*));
        for (size_t i = 0; i < result->rows; i++) {
            result->data[i] = calloc(result->cols, sizeof(int));
        }
    } else {
        // Non-root MPI processes do not allocate the full result matrix
        result->data = NULL;
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
            // Calculate the range of rows for each process
            size_t total_rows = A->num_rows;
            size_t rows_per_process = total_rows / size;
            size_t remainder = total_rows % size;

            size_t start_row = rank * rows_per_process + (rank < remainder ? rank : remainder);
            size_t end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);
            size_t local_rows = end_row - start_row;

            
            printf("Process %d: computing rows %zu to %zu (total %zu rows)\n", rank, start_row, end_row - 1, local_rows);

            // Allocate local data for each process
            int** local_data = malloc(local_rows * sizeof(int*));
            for (size_t i = 0; i < local_rows; i++) {
                local_data[i] = calloc(result->cols, sizeof(int));
            }

            // Perform local computation
            for (size_t i = start_row; i < end_row; i++) {
                for (size_t k = 0; k < A->row_sizes[i]; k++) {
                    int a_val = A->B[i][k];
                    size_t a_col = A->C[i][k];
                    for (size_t j = 0; j < B->row_sizes[a_col]; j++) {
                        size_t b_col = B->C[a_col][j];
                        int b_val = B->B[a_col][j];
                        local_data[i - start_row][b_col] += a_val * b_val;
                    }
                }
            }

            
            printf("Process %d: finished local computation\n", rank);

            // Gather the local results at the root process
            if (rank == 0) {
                // Copy local data to the result matrix
                for (size_t i = 0; i < local_rows; i++) {
                    memcpy(result->data[start_row + i], local_data[i], result->cols * sizeof(int));
                }

                // Receive data from other processes
                for (int proc = 1; proc < size; proc++) {
                    size_t proc_start_row = proc * rows_per_process + (proc < remainder ? proc : remainder);
                    size_t proc_end_row = proc_start_row + rows_per_process + (proc < remainder ? 1 : 0);
                    size_t proc_local_rows = proc_end_row - proc_start_row;

                    for (size_t i = 0; i < proc_local_rows; i++) {
                        
                        // printf("Root process: receiving data from process %d for row %zu\n", proc, proc_start_row + i);
                        MPI_Recv(result->data[proc_start_row + i], result->cols, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }

                
                printf("Root process: all data received and assembled\n");
            } else {
                // Send local data to the root process
                for (size_t i = 0; i < local_rows; i++) {
                    
                    // printf("Process %d: sending data for row %zu to root process\n", rank, start_row + i);
                    MPI_Send(local_data[i], result->cols, MPI_INT, 0, 0, MPI_COMM_WORLD);
                }

                
                printf("Process %d: all data sent to root process\n", rank);
            }

            // Free local data
            for (size_t i = 0; i < local_rows; i++) {
                free(local_data[i]);
            }
            free(local_data);

            // Non-root processes free the result structure
            if (rank != 0) {
                free(result);
                result = NULL;
            }
            break;
        }
    }

    // Stop timing
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
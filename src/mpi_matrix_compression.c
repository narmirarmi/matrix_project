#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <matrix_compression.h>


CompressedMatrix* compress_matrix_with_mpi(int** local_matrix, const size_t local_rows, const size_t cols, float density, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Allocate the CompressedMatrix structure
    CompressedMatrix* compressed = malloc(sizeof(CompressedMatrix));
    if (!compressed) {
        fprintf(stderr, "Process %d: Failed to allocate memory for CompressedMatrix\n", rank);
        return NULL;
    }

    compressed->num_rows = local_rows;
    compressed->num_cols = cols;

    size_t estimated_non_zero_per_row = (size_t)(cols * density) + 1;

    compressed->B = malloc(local_rows * sizeof(int*));
    compressed->C = malloc(local_rows * sizeof(int*));
    compressed->row_sizes = calloc(local_rows, sizeof(size_t));

    if (!compressed->B || !compressed->C || !compressed->row_sizes) {
        fprintf(stderr, "Process %d: Failed to allocate memory for compressed matrix arrays\n", rank);
        free_compressed_matrix(compressed);
        return NULL;
    }

    // Compress the local_matrix
    for (size_t i = 0; i < local_rows; i++) {
        compressed->B[i] = malloc(estimated_non_zero_per_row * sizeof(int));
        compressed->C[i] = malloc(estimated_non_zero_per_row * sizeof(int));

        if (!compressed->B[i] || !compressed->C[i]) {
            fprintf(stderr, "Process %d: Failed to allocate memory for row %zu\n", rank, i);
            continue;
        }

        size_t non_zero_count = 0;
        size_t current_capacity = estimated_non_zero_per_row;

        for (size_t j = 0; j < cols; j++) {
            if (local_matrix[i][j] != 0) {
                if (non_zero_count >= current_capacity) {
                    if (current_capacity > SIZE_MAX / 2) {
                        fprintf(stderr, "Cannot safely resize array for row %zu\n", i);
                        break;
                    }
                    size_t new_capacity = current_capacity * 2;
                    int* new_B = realloc(compressed->B[i], new_capacity * sizeof(int));
                    int* new_C = realloc(compressed->C[i], new_capacity * sizeof(int));

                    if (!new_B || !new_C) {
                        fprintf(stderr, "Failed to reallocate memory for row %zu\n", i);
                        break;
                    }

                    compressed->B[i] = new_B;
                    compressed->C[i] = new_C;
                    current_capacity = new_capacity;
                }
                compressed->B[i][non_zero_count] = local_matrix[i][j];
                compressed->C[i][non_zero_count] = j;
                non_zero_count++;
            }
        }

        if (non_zero_count == 0) {
            compressed->B[i][0] = 0;
            compressed->B[i][1] = 0;
            compressed->C[i][0] = 0;
            compressed->C[i][1] = 0;
            non_zero_count = 2;
        }

        int* trimmed_B = realloc(compressed->B[i], non_zero_count * sizeof(int));
        int* trimmed_C = realloc(compressed->C[i], non_zero_count * sizeof(int));

        if (trimmed_B) compressed->B[i] = trimmed_B;
        if (trimmed_C) compressed->C[i] = trimmed_C;

        compressed->row_sizes[i] = non_zero_count;
    }

    // No gathering; each process keeps its local compressed matrix
    return compressed;
}



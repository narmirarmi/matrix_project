#include "matrix_compression.h"
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <stdint.h>

CompressedMatrix* compress_matrix(int** matrix, const size_t rows, const size_t cols, float density) {
    CompressedMatrix* compressed = malloc(sizeof(CompressedMatrix));
    if (!compressed) {
        fprintf(stderr, "Failed to allocate memory for CompressedMatrix\n");
        return NULL;
    }
    compressed->num_rows = rows;
    compressed->num_cols = cols;

    // Estimate the number of useful elements per row (+1 for funsies)
    size_t estimated_non_zero_per_row = (size_t)(cols * density) + 1;

    // Allocate memory for B and C matrices
    compressed->B = malloc(rows * sizeof(int*));
    compressed->C = malloc(rows * sizeof(int*));
    compressed->row_sizes = calloc(rows, sizeof(size_t));

    if (!compressed->B || !compressed->C || !compressed->row_sizes) {
        fprintf(stderr, "Failed to allocate memory for compressed matrix arrays\n");
        free_compressed_matrix(compressed);
        return NULL;
    }

    // parallelize compression
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < rows; i++) {
            compressed->B[i] = malloc(estimated_non_zero_per_row * sizeof(int));
            compressed->C[i] = malloc(estimated_non_zero_per_row * sizeof(int));

            if (!compressed->B[i] || !compressed->C[i]) {
                fprintf(stderr, "Failed to allocate memory for row %zu\n", i);
                continue;
            }

            size_t non_zero_count = 0;
            size_t current_capacity = estimated_non_zero_per_row;

            for (size_t j = 0; j < cols; j++) {
                if (matrix[i][j] != 0) {
                    if (non_zero_count >= current_capacity) {
                        // Check overflow before reallocating
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
                    compressed->B[i][non_zero_count] = matrix[i][j];
                    compressed->C[i][non_zero_count] = j;
                    non_zero_count++;
                }
            }

            // If there are only zeros in this row, store two 0s
            if (non_zero_count == 0) {
                compressed->B[i][0] = 0;
                compressed->B[i][1] = 0;
                compressed->C[i][0] = 0;
                compressed->C[i][1] = 0;
                non_zero_count = 2;
            }

            // Trim excess memory
            int* trimmed_B = realloc(compressed->B[i], non_zero_count * sizeof(int));
            int* trimmed_C = realloc(compressed->C[i], non_zero_count * sizeof(int));

            if (trimmed_B) compressed->B[i] = trimmed_B;
            if (trimmed_C) compressed->C[i] = trimmed_C;

            compressed->row_sizes[i] = non_zero_count;
        }
    }

    return compressed;
}


void free_compressed_matrix(CompressedMatrix* compressed) {
    for (size_t i = 0; i < compressed->num_rows; i++) {
        free(compressed->B[i]);
        free(compressed->C[i]);
    }
    free(compressed->B);
    free(compressed->C);
    free(compressed->row_sizes);
    free(compressed);
}

void print_compressed_matrix(const CompressedMatrix* compressed) {
    printf("Matrix B (non-zero elements):\n");
    for (size_t i = 0; i < compressed->num_rows; i++) {
        for (size_t j = 0; j < compressed->row_sizes[i]; j++) {
            printf("%2d ", compressed->B[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrix C (column indices):\n");
    for (size_t i = 0; i < compressed->num_rows; i++) {
        for (size_t j = 0; j < compressed->row_sizes[i]; j++) {
            printf("%2d ", compressed->C[i][j]);
        }
        printf("\n");
    }
}
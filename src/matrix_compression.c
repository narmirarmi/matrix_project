#include "matrix_compression.h"
#include <stdlib.h>
#include <stdio.h>

CompressedMatrix* compress_matrix(int** matrix, const size_t rows, const size_t cols, float density) {
    CompressedMatrix* compressed = malloc(sizeof(CompressedMatrix));
    compressed->num_rows = rows;
    compressed->num_cols = cols;

    // Estimate the number useful elements per row (+1)
    size_t estimated_non_zero_per_row = (size_t)(cols * density) + 1;

    // Allocate memory for B and C matrices
    compressed->B = malloc(rows * sizeof(int*));
    compressed->C = malloc(rows * sizeof(int*));
    compressed->row_sizes = malloc(rows * sizeof(size_t));

    for (size_t i = 0; i < rows; i++) {
        compressed->B[i] = malloc(estimated_non_zero_per_row * sizeof(int));
        compressed->C[i] = malloc(estimated_non_zero_per_row * sizeof(int));

        size_t non_zero_count = 0;
        for (size_t j = 0; j < cols; j++) {
            if (matrix[i][j] != 0) {
                if (non_zero_count >= estimated_non_zero_per_row) {
                    // Reallocate if our estimate was too low
                    estimated_non_zero_per_row *= 2;
                    compressed->B[i] = realloc(compressed->B[i], estimated_non_zero_per_row * sizeof(int));
                    compressed->C[i] = realloc(compressed->C[i], estimated_non_zero_per_row * sizeof(int));
                }
                compressed->B[i][non_zero_count] = matrix[i][j];
                compressed->C[i][non_zero_count] = j;
                non_zero_count++;
            }
        }

        // If there are no non-zero elements in this row, store two consecutive 0s
        if (non_zero_count == 0) {
            compressed->B[i][0] = 0;
            compressed->B[i][1] = 0;
            compressed->C[i][0] = 0;
            compressed->C[i][1] = 0;
            non_zero_count = 2;
        }

        // Trim excess memory
        compressed->B[i] = realloc(compressed->B[i], non_zero_count * sizeof(int));
        compressed->C[i] = realloc(compressed->C[i], non_zero_count * sizeof(int));
        compressed->row_sizes[i] = non_zero_count;
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
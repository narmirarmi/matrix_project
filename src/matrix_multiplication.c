#include "matrix_multiplication.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "timing.h"

DenseMatrix* multiply_matrices(const CompressedMatrix* A, const CompressedMatrix* B) {
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

    // Perform matrix multiplication
    for (size_t i = 0; i < A->num_rows; i++) {
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
#ifndef MATRIX_MULTIPLICATION_H
#define MATRIX_MULTIPLICATION_H

#include "matrix_compression.h"
#include <stddef.h>

typedef enum {
    MULT_SEQUENTIAL,
    MULT_OMP,
    MULT_MPI,
} ScheduleType;

typedef struct {
    int** data;
    size_t rows;
    size_t cols;
} DenseMatrix;

// Function Prototypes
// FUnction to multiply two compressed matrices and return a dense matrix
DenseMatrix* multiply_matrices(const CompressedMatrix* A, const CompressedMatrix* B, ScheduleType schedule_type);

// Function to free a dense matrix
void free_dense_matrix(DenseMatrix* matrix);

// Function to print a dense matrix
void print_dense_matrix(const DenseMatrix* matrix);

#endif // MATRIX_MULTIPLICATION_H
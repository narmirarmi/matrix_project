#ifndef MATRIX_COMPRESSION_H
#define MATRIX_COMPRESSION_H

#include <stddef.h>

typedef struct {
    int** B;  // Matrix B storing non-zero elements
    int** C;  // Matrix C storing column indices
    size_t* row_sizes;  // Actual number of elements in each row
    size_t num_rows;
    size_t num_cols;
} CompressedMatrix;

// Function prototypes
CompressedMatrix* compress_matrix(int** matrix, size_t rows, size_t cols, float density);
void free_compressed_matrix(CompressedMatrix* compressed);
void print_compressed_matrix(const CompressedMatrix* compressed);

#endif // MATRIX_COMPRESSION_H
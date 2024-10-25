#ifndef MPI_MATRIX_COMPRESSION_H
#define MPI_MATRIX_COMPRESSION_H

#include <stddef.h>
#include <mpi.h>
#include "matrix_compression.h"

// Function prototype for the MPI version of compress_matrix
CompressedMatrix* compress_matrix_with_mpi(int** matrix, size_t rows, size_t cols, float density);

#endif // MPI_MATRIX_COMPRESSION_H
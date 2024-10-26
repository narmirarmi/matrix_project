#ifndef MPI_MATRIX_COMPRESSION_H
#define MPI_MATRIX_COMPRESSION_H

#include <stddef.h>
#include <mpi.h>
#include "matrix_compression.h"

// Function prototype for the MPI version of compress_matrix
CompressedMatrix* compress_matrix_with_mpi(int** local_matrix, const size_t local_rows, const size_t cols, float density, MPI_Comm comm);
#endif // MPI_MATRIX_COMPRESSION_H
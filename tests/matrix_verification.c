#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix_compression.h"
#include "matrix_multiplication.h"

// Function to verify matrix multiplication
int verify_matrix_multiplication(int** A, int** B, DenseMatrix* result, int rows_a, int cols_a, int cols_b) {
    for (int i = 0; i < rows_a; i++) {
        for (int j = 0; j < cols_b; j++) {
            int expected = 0;
            for (int k = 0; k < cols_a; k++) {
                expected += A[i][k] * B[k][j];
            }
            if (result->data[i][j] != expected) {
                printf("Mismatch at position (%d, %d): Expected %d, Got %d\n", i, j, expected, result->data[i][j]);
                return 0;  // Verification failed
            }
        }
    }
    return 1;  // Verification successful
}

// Function to generate a small test case
void test_small_case() {
    int rows_a = 3, cols_a = 4, cols_b = 2;
    int A[3][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    int B[4][2] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    
    // Convert to required format
    int** matrix_a = malloc(rows_a * sizeof(int*));
    int** matrix_b = malloc(cols_a * sizeof(int*));
    for (int i = 0; i < rows_a; i++) matrix_a[i] = malloc(cols_a * sizeof(int));
    for (int i = 0; i < cols_a; i++) matrix_b[i] = malloc(cols_b * sizeof(int));
    
    for (int i = 0; i < rows_a; i++)
        for (int j = 0; j < cols_a; j++)
            matrix_a[i][j] = A[i][j];
    
    for (int i = 0; i < cols_a; i++)
        for (int j = 0; j < cols_b; j++)
            matrix_b[i][j] = B[i][j];
    
    CompressedMatrix* compressed_a = compress_matrix(matrix_a, rows_a, cols_a, 1.0);
    CompressedMatrix* compressed_b = compress_matrix(matrix_b, cols_a, cols_b, 1.0);
    
    DenseMatrix* result = multiply_matrices(compressed_a, compressed_b);
    
    if (verify_matrix_multiplication(matrix_a, matrix_b, result, rows_a, cols_a, cols_b)) {
        printf("Small case verification successful!\n");
    } else {
        printf("Small case verification failed.\n");
    }
    
    // Clean up
    for (int i = 0; i < rows_a; i++) free(matrix_a[i]);
    for (int i = 0; i < cols_a; i++) free(matrix_b[i]);
    free(matrix_a);
    free(matrix_b);
    free_compressed_matrix(compressed_a);
    free_compressed_matrix(compressed_b);
    free_dense_matrix(result);
}

int main() {
    test_small_case();
    return 0;
}
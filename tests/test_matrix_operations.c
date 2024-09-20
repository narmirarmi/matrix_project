#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix_generation.h"
#include "matrix_compression.h"
#include "matrix_multiplication.h"

// Function to generate a random dense matrix
int** generate_random_matrix(int rows, int cols, float density) {
    int** matrix = allocateMatrix(rows, cols);
    initialiseMatrix(matrix, rows, cols, density);
    return matrix;
}

// Function to test matrix compression
void test_matrix_compression(int rows, int cols, float density) {
    printf("Testing matrix compression (%dx%d, density: %.2f)\n", rows, cols, density);
    int** dense_matrix = generate_random_matrix(rows, cols, density);
    CompressedMatrix* compressed = compress_matrix(dense_matrix, rows, cols, density);
    
    printf("Original matrix:\n");
    printMatrix(dense_matrix, rows, cols);
    
    printf("Compressed matrix:\n");
    print_compressed_matrix(compressed);
    
    // Clean up
    freeMatrix(dense_matrix, rows);
    free_compressed_matrix(compressed);
}

// Function to test matrix multiplication
void test_matrix_multiplication(int rows_a, int cols_a, int cols_b, float density) {
    printf("Testing matrix multiplication (%dx%d * %dx%d, density: %.2f)\n", 
           rows_a, cols_a, cols_a, cols_b, density);
    
    int** dense_a = generate_random_matrix(rows_a, cols_a, density);
    int** dense_b = generate_random_matrix(cols_a, cols_b, density);
    
    CompressedMatrix* compressed_a = compress_matrix(dense_a, rows_a, cols_a, density);
    CompressedMatrix* compressed_b = compress_matrix(dense_b, cols_a, cols_b, density);
    
    DenseMatrix* result = multiply_matrices(compressed_a, compressed_b);
    
    printf("Result matrix:\n");
    print_dense_matrix(result);
    
    // Clean up
    freeMatrix(dense_a, rows_a);
    freeMatrix(dense_b, cols_a);
    free_compressed_matrix(compressed_a);
    free_compressed_matrix(compressed_b);
    free_dense_matrix(result);
}

int main() {
    srand(time(NULL));  // Seed the random number generator
    
    // Test small matrices
    test_matrix_compression(5, 5, 0.5);
    test_matrix_multiplication(3, 4, 5, 0.5);
    
    // Test medium-sized matrices
    test_matrix_compression(100, 100, 0.1);
    test_matrix_multiplication(50, 60, 70, 0.1);
    
    // Test large matrices (uncomment when ready for larger tests)
    test_matrix_compression(1000, 1000, 0.01);
    test_matrix_multiplication(500, 600, 700, 0.01);
    //
    // Test very large matrices (as per project requirements)
    // Note: These may take a long time to run
    test_matrix_compression(100000, 100000, 0.01);
    test_matrix_multiplication(100000, 100000, 100000, 0.01);
    // test_matrix_compression(100000, 100000, 0.02);
    // test_matrix_multiplication(100000, 100000, 100000, 0.02);
    // test_matrix_compression(100000, 100000, 0.05);
    // test_matrix_multiplication(100000, 100000, 100000, 0.05);
    
    return 0;
}
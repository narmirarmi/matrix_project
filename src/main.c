#include "matrix_generation.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    // Seed the random number generator
    srand(time(NULL));

    // Define sparsity (e.g., 0.01 for 1% non-zero elements)
    float sparsity = 1;

    // Allocate and initialize matrix
    int** matrix = allocateMatrix(ROWS, COLS);
    initialiseMatrix(matrix, ROWS, COLS, sparsity);

    // Print the matrix
    printf("Generated Matrix (Sparsity: %.2f):\n", sparsity);
    printMatrix(matrix, ROWS, COLS);

    // Free the matrix
    freeMatrix(matrix, ROWS);

    return 0;
}

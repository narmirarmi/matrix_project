#include "matrix_generation.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int setCellValue(const float sparsity) {
    // Generate a random float between 0 and 1
    const float random_float = (float)rand() / RAND_MAX;

    // If the random float is less than the sparsity, set a non-zero value
    if (random_float < sparsity) {
        // Return int between 1 and 10
        return (rand() % 10) + 1;
    } else {
        return 0;
    }
}


int **allocateMatrix(const int rows, const int cols) {
    int** matrix = (int**) malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*) malloc(cols * sizeof(int*));
    }
  return matrix;
}


void initialiseMatrix(int** matrix, const int rows, const int cols, const float sparsity) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) {
            matrix[i][j] = setCellValue(sparsity);
        }
    }

}

void freeMatrix(int** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void printMatrix(int** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%2d ", matrix[i][j]);
        }
        printf("\n");
    }
}

#ifndef MATRIX_GENERATION_H
#define MATRIX_GENERATION_H

#define ROWS 15
#define COLS 15

// Function prototypes
int** allocateMatrix(int rows, int cols);
void initialiseMatrix(int** matrix, int rows, int cols, float sparsity);
void freeMatrix(int** matrix, int rows);
void printMatrix(int** matrix, int rows, int cols);
int setCellValue(float sparsity);

#endif // MATRIX_GENERATION_H
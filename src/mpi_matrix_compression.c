#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <matrix_compression.h>

CompressedMatrix* compress_local_matrix(int* local_matrix, size_t local_rows, size_t cols, float density) {
    CompressedMatrix* compressed = malloc(sizeof(CompressedMatrix));
    if (!compressed) {
        fprintf(stderr, "Failed to allocate memory for CompressedMatrix\n");
        return NULL;
    }
    compressed->num_rows = local_rows;
    compressed->num_cols = cols;

    size_t estimated_non_zero_per_row = (size_t)(cols * density) + 1;

    compressed->B = malloc(local_rows * sizeof(int*));
    compressed->C = malloc(local_rows * sizeof(int*));
    compressed->row_sizes = calloc(local_rows, sizeof(size_t));

    if (!compressed->B || !compressed->C || !compressed->row_sizes) {
        fprintf(stderr, "Failed to allocate memory for compressed matrix arrays\n");
        free_compressed_matrix(compressed);
        return NULL;
    }

    for (size_t i = 0; i < local_rows; i++) {
        compressed->B[i] = malloc(estimated_non_zero_per_row * sizeof(int));
        compressed->C[i] = malloc(estimated_non_zero_per_row * sizeof(int));

        if (!compressed->B[i] || !compressed->C[i]) {
            fprintf(stderr, "Failed to allocate memory for row %zu\n", i);
            continue;
        }

        size_t non_zero_count = 0;
        size_t current_capacity = estimated_non_zero_per_row;

        for (size_t j = 0; j < cols; j++) {
            int value = local_matrix[i * cols + j];
            if (value != 0) {
                if (non_zero_count >= current_capacity) {
                    size_t new_capacity = current_capacity * 2;
                    int* new_B = realloc(compressed->B[i], new_capacity * sizeof(int));
                    int* new_C = realloc(compressed->C[i], new_capacity * sizeof(int));

                    if (!new_B || !new_C) {
                        fprintf(stderr, "Failed to reallocate memory for row %zu\n", i);
                        break;
                    }

                    compressed->B[i] = new_B;
                    compressed->C[i] = new_C;
                    current_capacity = new_capacity;
                }
                compressed->B[i][non_zero_count] = value;
                compressed->C[i][non_zero_count] = j;
                non_zero_count++;
            }
        }

        if (non_zero_count == 0) {
            compressed->B[i][0] = 0;
            compressed->B[i][1] = 0;
            compressed->C[i][0] = 0;
            compressed->C[i][1] = 0;
            non_zero_count = 2;
        } else {
            int* trimmed_B = realloc(compressed->B[i], non_zero_count * sizeof(int));
            int* trimmed_C = realloc(compressed->C[i], non_zero_count * sizeof(int));

            if (trimmed_B) compressed->B[i] = trimmed_B;
            if (trimmed_C) compressed->C[i] = trimmed_C;
        }

        compressed->row_sizes[i] = non_zero_count;
    }

    return compressed;
}

CompressedMatrix* compress_matrix_with_mpi(int** matrix, const size_t rows, const size_t cols, float density) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // root has FLAT matrix
    int* flat_matrix = NULL;
    if (rank == 0) {
        flat_matrix = malloc(rows * cols * sizeof(int));
        if (!flat_matrix) {
            fprintf(stderr, "Failed to allocate memory for flat_matrix\n");
            return NULL;
        }
        for (size_t i = 0; i < rows; i++) {
            memcpy(&flat_matrix[i * cols], matrix[i], cols * sizeof(int));
        }
    }

    // compute locally at each process
    size_t* local_rows_array = malloc(size * sizeof(size_t));
    int* sendcounts = malloc(size * sizeof(int));
    int* displs = malloc(size * sizeof(int));

    if (!local_rows_array || !sendcounts || !displs) {
        fprintf(stderr, "Failed to allocate memory for local_rows_array/sendcounts/displs\n");
        if (flat_matrix) free(flat_matrix);
        free(local_rows_array);
        free(sendcounts);
        free(displs);
        return NULL;
    }

    size_t sum_local_rows = 0;
    for (int r = 0; r < size; r++) {
        size_t local_rows = rows / size + (r < (rows % size) ? 1 : 0);
        local_rows_array[r] = local_rows;
        sendcounts[r] = local_rows * cols;
        displs[r] = sum_local_rows * cols;
        sum_local_rows += local_rows;
    }

    size_t local_rows = local_rows_array[rank];

    int* local_matrix = malloc(local_rows * cols * sizeof(int));
    if (!local_matrix) {
        fprintf(stderr, "Failed to allocate memory for local_matrix\n");
        if (flat_matrix) free(flat_matrix);
        free(local_rows_array);
        free(sendcounts);
        free(displs);
        return NULL;
    }

    MPI_Scatterv(flat_matrix, sendcounts, displs, MPI_INT, local_matrix, local_rows * cols, MPI_INT, 0, MPI_COMM_WORLD);

    // compress localy at each process
    CompressedMatrix* local_compressed = compress_local_matrix(local_matrix, local_rows, cols, density);

    // serialize data so we can send it to root
    size_t total_size = sizeof(size_t); // number of local_rows
    for (size_t i = 0; i < local_rows; i++) {
        size_t row_size = local_compressed->row_sizes[i];
        total_size += sizeof(size_t); // row_size
        total_size += row_size * sizeof(int); // B[i]
        total_size += row_size * sizeof(int); // C[i]
    }

    char* buffer = malloc(total_size);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate serialization buffer\n");
        free(local_matrix);
        free_compressed_matrix(local_compressed);
        if (flat_matrix) free(flat_matrix);
        free(local_rows_array);
        free(sendcounts);
        free(displs);
        return NULL;
    }

    char* ptr = buffer;
    memcpy(ptr, &local_rows, sizeof(size_t));
    ptr += sizeof(size_t);

    for (size_t i = 0; i < local_rows; i++) {
        size_t row_size = local_compressed->row_sizes[i];
        memcpy(ptr, &row_size, sizeof(size_t));
        ptr += sizeof(size_t);

        memcpy(ptr, local_compressed->B[i], row_size * sizeof(int));
        ptr += row_size * sizeof(int);

        memcpy(ptr, local_compressed->C[i], row_size * sizeof(int));
        ptr += row_size * sizeof(int);
    }

    if (rank == 0) {
        size_t* total_sizes = malloc(size * sizeof(size_t));
        char** recv_buffers = malloc(size * sizeof(char*));
        if (!total_sizes || !recv_buffers) {
            fprintf(stderr, "Failed to allocate total_sizes or recv_buffers\n");
            free(buffer);
            free(local_matrix);
            free_compressed_matrix(local_compressed);
            if (flat_matrix) free(flat_matrix);
            free(local_rows_array);
            free(sendcounts);
            free(displs);
            return NULL;
        }

        total_sizes[0] = total_size;
        recv_buffers[0] = buffer;

        for (int r = 1; r < size; r++) {
            MPI_Recv(&total_sizes[r], 1, MPI_UNSIGNED_LONG_LONG, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recv_buffers[r] = malloc(total_sizes[r]);
            MPI_Recv(recv_buffers[r], total_sizes[r], MPI_BYTE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // root constructs from received data
        CompressedMatrix* compressed = malloc(sizeof(CompressedMatrix));
        if (!compressed) {
            fprintf(stderr, "Failed to allocate CompressedMatrix\n");
            free(total_sizes);
            for (int r = 0; r < size; r++) {
                free(recv_buffers[r]);
            }
            free(recv_buffers);
            free(buffer);
            free(local_matrix);
            free_compressed_matrix(local_compressed);
            if (flat_matrix) free(flat_matrix);
            free(local_rows_array);
            free(sendcounts);
            free(displs);
            return NULL;
        }

        compressed->num_rows = rows;
        compressed->num_cols = cols;
        compressed->B = malloc(rows * sizeof(int*));
        compressed->C = malloc(rows * sizeof(int*));
        compressed->row_sizes = malloc(rows * sizeof(size_t));

        if (!compressed->B || !compressed->C || !compressed->row_sizes) {
            fprintf(stderr, "Failed to allocate compressed matrix arrays\n");
            free_compressed_matrix(compressed);
            free(total_sizes);
            for (int r = 0; r < size; r++) {
                free(recv_buffers[r]);
            }
            free(recv_buffers);
            free(buffer);
            free(local_matrix);
            free_compressed_matrix(local_compressed);
            if (flat_matrix) free(flat_matrix);
            free(local_rows_array);
            free(sendcounts);
            free(displs);
            return NULL;
        }

        size_t row_index = 0;
        for (int r = 0; r < size; r++) {
            char* ptr = recv_buffers[r];
            size_t proc_local_rows;
            memcpy(&proc_local_rows, ptr, sizeof(size_t));
            ptr += sizeof(size_t);

            for (size_t i = 0; i < proc_local_rows; i++) {
                size_t row_size;
                memcpy(&row_size, ptr, sizeof(size_t));
                ptr += sizeof(size_t);

                compressed->row_sizes[row_index] = row_size;

                compressed->B[row_index] = malloc(row_size * sizeof(int));
                memcpy(compressed->B[row_index], ptr, row_size * sizeof(int));
                ptr += row_size * sizeof(int);

                compressed->C[row_index] = malloc(row_size * sizeof(int));
                memcpy(compressed->C[row_index], ptr, row_size * sizeof(int));
                ptr += row_size * sizeof(int);

                row_index++;
            }
            free(recv_buffers[r]);
        }

        free(total_sizes);
        free(recv_buffers);
        free(buffer);
        free(local_matrix);
        free_compressed_matrix(local_compressed);
        if (flat_matrix) free(flat_matrix);
        free(local_rows_array);
        free(sendcounts);
        free(displs);
        return compressed;
    } else {
        // non-root
        MPI_Send(&total_size, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
        MPI_Send(buffer, total_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);

        free(buffer);
        free(local_matrix);
        free_compressed_matrix(local_compressed);
        if (flat_matrix) free(flat_matrix);
        free(local_rows_array);
        free(sendcounts);
        free(displs);
        return NULL;
    }
}

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {

    // init MPI
    MPI_Init(NULL, NULL);

    // get num processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // get processor rank
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // get processor name
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Get the name of the processor
    printf("Hello world from processor %s, rank %d out of %d processors\n",
        processor_name, world_rank, world_size );

    // Finalize MPI environment
    MPI_Finalize();

    return 0;
}
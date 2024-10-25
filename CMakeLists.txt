cmake_minimum_required(VERSION 3.29)
project(matrix_project C)

set(CMAKE_C_STANDARD 11)

find_package(MPI REQUIRED)
find_package(OpenMP)

# Set MPI compiler wrapper
set(CMAKE_C_COMPILER ${MPI_C_COMPILER})

# Add include directory to include path
include_directories(include)

# Define source files for reuse
set(COMMON_SOURCES
    src/matrix_generation.c
    src/matrix_compression.c
    src/matrix_multiplication.c
)

# Create the main executable
add_executable(matrix_project
    src/main.c
    ${COMMON_SOURCES}
)

# Create the test executable
add_executable(run_tests
    tests/test_parallel_matrices_varying_parallelisation.c
    ${COMMON_SOURCES}
)

add_executable(verify_multiplication
    ${COMMON_SOURCES}
)

# Set working directory for all executables
set_target_properties(matrix_project run_tests verify_multiplication
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)


add_custom_target(test
    COMMAND mpirun -np 4 $<TARGET_FILE:run_tests>
    DEPENDS run_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# If using OpenMP
find_package(OpenMP)

# If using MPI
find_package(MPI REQUIRED)

if(OpenMP_C_FOUND)
    set(EXECUTABLES matrix_project run_tests verify_multiplication)
    foreach(EXEC ${EXECUTABLES})
        target_link_libraries(${EXEC} PUBLIC OpenMP::OpenMP_C)
    endforeach()
endif()

# Link MPI to all executables
foreach(EXEC matrix_project run_tests verify_multiplication)
    target_link_libraries(${EXEC} PUBLIC MPI::MPI_C)
    # Ensure MPI compile definitions are added
    target_compile_definitions(${EXEC} PRIVATE OMPI_SKIP_MPICXX)
endforeach()
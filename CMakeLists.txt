cmake_minimum_required(VERSION 3.29)
project(matrix_project C)

set(CMAKE_C_STANDARD 11)

# Add include directory to include path
include_directories(include)

# Create the main executable
add_executable(matrix_project
        src/main.c
        src/matrix_generation.c
        src/matrix_compression.c
        src/matrix_multiplication.c
        include/timing.h
)

# Create the test executable
add_executable(run_tests
        tests/test_parallel_matrices.c
        src/matrix_generation.c
        src/matrix_compression.c
        src/matrix_multiplication.c
)

add_executable(verify_multiplication
        src/matrix_generation.c
        src/matrix_compression.c
        src/matrix_multiplication.c
)

# Optional: Add a custom target to run tests
add_custom_target(test
        COMMAND run_tests
        DEPENDS run_tests
)

# If using OpenMP
find_package(OpenMP)

# If using MPI
find_package(MPI REQUIRED)

if(OpenMP_C_FOUND)
    target_link_libraries(matrix_project PUBLIC OpenMP::OpenMP_C)
    target_link_libraries(run_tests PUBLIC OpenMP::OpenMP_C)
endif()

target_link_libraries(matrix_project PRIVATE MPI::MPI_C)

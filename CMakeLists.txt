cmake_minimum_required(VERSION 3.29)
project(matrix_project C)

set(CMAKE_C_STANDARD 11)

# Set up OpenMP for macOS
if(APPLE)
    set(CMAKE_C_COMPILER clang)

    # Properly set up OpenMP paths using Homebrew
    execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    set(OpenMP_C_INCLUDE_DIRS "${LIBOMP_PREFIX}/include")

    # Ensure CMake can find OpenMP
    include_directories(${OpenMP_C_INCLUDE_DIRS})
    link_directories(${LIBOMP_PREFIX}/lib)
endif()

# Add include directory to include path
include_directories(include)

# If using MPI
find_package(MPI REQUIRED)

# If using OpenMP
find_package(OpenMP REQUIRED)

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
        tests/test_parallel_matrices_varying_parallelisation.c
        src/matrix_generation.c
        src/matrix_compression.c
        src/matrix_multiplication.c
)

add_executable(verify_multiplication
        src/matrix_generation.c
        src/matrix_compression.c
        src/matrix_multiplication.c
)


# Add option to specify number of processes
set(MPI_NUM_PROCESSES ${NUM_CORES} CACHE STRING "Number of MPI processes to use")

# Optional: Add a custom target to run tests
add_custom_target(test
        COMMAND run_tests
        DEPENDS run_tests
)


# Link libraries for each target
target_link_libraries(matrix_project PRIVATE
        OpenMP::OpenMP_C
        MPI::MPI_C
)

target_link_libraries(run_tests PRIVATE
        OpenMP::OpenMP_C
        MPI::MPI_C
)

target_link_libraries(verify_multiplication PRIVATE
        OpenMP::OpenMP_C
        MPI::MPI_C
)

# For macOS, add compiler and linker flags
if(APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}")
endif()

# Update run_tests target to use generated hostfile
add_custom_target(run_tests_mpi
        COMMAND ${MPIEXEC_EXECUTABLE}
        -np 4
        $<TARGET_FILE:run_tests>
        -m
        DEPENDS run_tests
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running tests with MPI locally"
)

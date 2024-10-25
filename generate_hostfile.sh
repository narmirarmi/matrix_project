#!/bin/bash

# Get system
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     NUM_CORES=$(sysctl -n hw.ncpu)  # For macO
    Darwin*)    NUM_CORES=$(nproc)
    *)          machine="UNKNOWN:${unameOut}"
esac

# Create or overwrite hostfile
echo "# Auto-generated hostfile" > hostfile.txt
echo "localhost slots=$NUM_CORES max_slots=$NUM_CORES" >> hostfile.txt

# Optional: Add CPU binding
echo "# CPU binding configuration" >> hostfile.txt
CPU_LIST=$(seq -s ',' 0 $((NUM_CORES-1)))
echo "localhost slots=$NUM_CORES max_slots=$NUM_CORES cpu_ids=$CPU_LIST" >> hostfile.txt

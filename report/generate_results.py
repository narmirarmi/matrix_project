import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def parse_performance_file(file_path):
    """Parse a single performance CSV file and extract relevant information."""
    try:
        # Check if file exists and has content
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"Skipping empty or non-existent file: {file_path}")
            return None

        # First read metadata
        with open(file_path, 'r') as f:
            lines = f.readlines()

            # Extract metadata
            try:
                size_match = re.search(r'(\d+) x (\d+)', lines[0])
                density = float(re.search(r'Density: ([\d.]+)', lines[2]).group(1))
                parallelisation = re.search(r'Parallelisation: (\w+)', lines[3]).group(1)
                matrix_size = int(size_match.group(1))  # Assuming square matrices
            except (AttributeError, ValueError, IndexError) as e:
                print(f"Error parsing metadata in {file_path}: {e}")
                return None

        # Read the CSV part starting from the header row
        try:
            df = pd.read_csv(file_path, skiprows=5)  # Skip the metadata lines

            if df.empty:
                print(f"No performance data found in {file_path}")
                return None

            return {
                'matrix_size': matrix_size,
                'density': density,
                'parallelisation': parallelisation,
                'cpu_time': df['CPU Time (s)'].iloc[0],
                'wall_time': df['Wall Clock Time (s)'].iloc[0]
            }

        except pd.errors.EmptyDataError:
            print(f"No data found in {file_path}")
            return None
        except Exception as e:
            print(f"Error reading CSV data from {file_path}: {e}")
            return None

    except Exception as e:
        print(f"Unexpected error processing {file_path}: {e}")
        return None

def collect_performance_data(logs_dir):
    """Traverse the logs directory and collect all performance data."""
    results = []

    for root, dirs, files in os.walk(logs_dir):
        # Remove 'old' from dirs list to prevent os.walk from traversing into it
        if 'old' in dirs:
            dirs.remove('old')

        for file in files:
            if file.endswith('.csv') and 'performance' in file:
                file_path = os.path.join(root, file)
                result = parse_performance_file(file_path)
                if result is not None:
                    results.append(result)

    df = pd.DataFrame(results)
    if df.empty:
        print("No valid performance data files found!")
    else:
        print(f"Successfully parsed {len(df)} performance files")

    return df

def set_plot_style():
    """Set consistent style for all plots"""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10

def plot_time_vs_size(df, output_dir):
    """Plot execution time vs matrix size for each parallelisation strategy."""
    for density in df['density'].unique():
        plt.figure()
        df_density = df[df['density'] == density]

        # Create color palette
        colors = sns.color_palette("husl", n_colors=len(df_density['parallelisation'].unique()))

        for i, parallel_type in enumerate(df_density['parallelisation'].unique()):
            df_filtered = df_density[df_density['parallelisation'] == parallel_type]
            plt.plot(df_filtered['matrix_size'],
                     df_filtered['wall_time'],
                     marker='o',
                     color=colors[i],
                     label=parallel_type.capitalize())

        plt.title(f'Matrix Multiplication Performance (Density: {density})')
        plt.xlabel('Matrix Size (N×N)')
        plt.ylabel('Wall Clock Time (seconds)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'time_vs_size_density_{density}.png'))
        plt.close()

def plot_max_size_by_density(df, output_dir):
    """Plot maximum matrix size achieved within 10 minutes for each strategy and density."""
    # Filter for results under 10 minutes (600 seconds)
    df_filtered = df[df['wall_time'] <= 600]

    plt.figure()
    colors = sns.color_palette("husl", n_colors=len(df_filtered['parallelisation'].unique()))

    for i, parallel_type in enumerate(df_filtered['parallelisation'].unique()):
        df_parallel = df_filtered[df_filtered['parallelisation'] == parallel_type]

        sizes_by_density = []
        densities = sorted(df_parallel['density'].unique())

        for density in densities:
            max_size = df_parallel[df_parallel['density'] == density]['matrix_size'].max()
            sizes_by_density.append(max_size)

        plt.plot(densities, sizes_by_density, marker='o', color=colors[i],
                 label=parallel_type.capitalize())

    plt.title('Maximum Matrix Size Within 10 Minutes')
    plt.xlabel('Matrix Density')
    plt.ylabel('Maximum Matrix Size (N)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'max_size_by_density.png'))
    plt.close()

def plot_speedup(df, output_dir):
    """Plot speedup relative to sequential execution."""
    for density in df['density'].unique():
        print(f"\nProcessing density: {density}")
        df_density = df[df['density'] == density]

        # Get sequential times for this density
        sequential_df = df_density[df_density['parallelisation'] == 'sequential']
        if sequential_df.empty:
            print(f"No sequential data found for density {density}")
            continue

        # Create a dictionary of sequential times by matrix size for easier lookup
        sequential_times = dict(zip(sequential_df['matrix_size'], sequential_df['wall_time']))
        print(f"Sequential times: {sequential_times}")

        plt.figure()
        colors = sns.color_palette("husl", n_colors=len(df_density['parallelisation'].unique()) - 1)

        color_idx = 0
        for parallel_type in sorted(df_density['parallelisation'].unique()):
            if parallel_type == 'sequential':
                continue

            df_parallel = df_density[df_density['parallelisation'] == parallel_type]
            print(f"\nProcessing {parallel_type}")

            # Create lists for valid sizes and their corresponding speedups
            valid_sizes = []
            speedups = []

            # Only process sizes that exist in both sequential and parallel results
            for size in sorted(df_parallel['matrix_size'].unique()):
                if size in sequential_times:
                    seq_time = sequential_times[size]
                    parallel_time = df_parallel[df_parallel['matrix_size'] == size]['wall_time'].iloc[0]
                    speedup = seq_time / parallel_time

                    print(f"Size {size}: Sequential={seq_time:.2f}s, {parallel_type}={parallel_time:.2f}s, Speedup={speedup:.2f}x")

                    valid_sizes.append(size)
                    speedups.append(speedup)

            if valid_sizes:  # Only plot if we have data
                print(f"Plotting data points: sizes={valid_sizes}, speedups={speedups}")
                plt.plot(valid_sizes, speedups, marker='o', color=colors[color_idx],
                         label=f"{parallel_type.capitalize()} (vs Sequential)")
                color_idx += 1
            else:
                print(f"No valid comparison points found for {parallel_type}")

        plt.title(f'Speedup vs Sequential (Density: {density})')
        plt.xlabel('Matrix Size (N×N)')
        plt.ylabel('Speedup Factor')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f'speedup_density_{density}.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved speedup plot to {output_path}")

def plot_time_vs_size(df, output_dir):
    """Plot execution time vs matrix size for each parallelisation strategy."""
    for density in df['density'].unique():
        plt.figure(figsize=(12, 8))
        df_density = df[df['density'] == density]

        # Create color palette
        parallel_types = sorted(df_density['parallelisation'].unique())
        colors = sns.color_palette("husl", n_colors=len(parallel_types))

        for i, parallel_type in enumerate(parallel_types):
            df_filtered = df_density[df_density['parallelisation'] == parallel_type]
            # Sort by matrix size for proper line plotting
            df_filtered = df_filtered.sort_values('matrix_size')

            plt.plot(df_filtered['matrix_size'],
                     df_filtered['wall_time'],
                     marker='o',
                     color=colors[i],
                     label=parallel_type.capitalize())

        plt.title(f'Matrix Multiplication Performance (Density: {density})')
        plt.xlabel('Matrix Size (N×N)')
        plt.ylabel('Wall Clock Time (seconds)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f'time_vs_size_density_{density}.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved time vs size plot to {output_path}")

def plot_density_impact(df, output_dir):
    """Plot how density affects execution time for each parallelisation strategy."""
    plt.figure(figsize=(12, 8))

    # Sort parallel types for consistent coloring
    parallel_types = sorted(df['parallelisation'].unique())
    colors = sns.color_palette("husl", n_colors=len(parallel_types))

    for i, parallel_type in enumerate(parallel_types):
        df_parallel = df[df['parallelisation'] == parallel_type]

        # For each matrix size, plot time vs density
        for size in sorted(df_parallel['matrix_size'].unique()):
            df_size = df_parallel[df_parallel['matrix_size'] == size]
            # Sort by density for proper line plotting
            df_size = df_size.sort_values('density')

            plt.plot(df_size['density'],
                     df_size['wall_time'],
                     marker='o',
                     color=colors[i],
                     label=f'{parallel_type.capitalize()} (N={size})')

    plt.title('Impact of Matrix Density on Execution Time')
    plt.xlabel('Density')
    plt.ylabel('Wall Clock Time (seconds)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'density_impact.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved density impact plot to {output_path}")

def main():
    # Set paths
    logs_dir = "/home/narmirarmi/CLionProjects/matrix_project/cmake-build-debug/logs"
    output_dir = "performance_plots"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set plot style
    set_plot_style()

    # Collect and process data
    print("Collecting performance data...")
    df = collect_performance_data(logs_dir)

    if df.empty:
        print("No data found in the logs directory!")
        return

    print(f"Found {len(df)} results")
    print("\nGenerating plots...")

    # Generate plots
    plot_time_vs_size(df, output_dir)
    plot_max_size_by_density(df, output_dir)
    plot_speedup(df, output_dir)
    plot_density_impact(df, output_dir)

    # Save processed data
    df.to_csv(os.path.join(output_dir, 'processed_results.csv'), index=False)
    print(f"\nPlots have been saved to: {output_dir}")
    print(f"Processed data saved to: {os.path.join(output_dir, 'processed_results.csv')}")

if __name__ == "__main__":
    main()
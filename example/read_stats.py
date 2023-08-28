import os
import pandas as pd
import seaborn as sns
import numpy as np
import sys

# This function parses the directory path and returns the benchmark and
# row to which the runtime belongs to
def parse_path(subdir):
    tokens = subdir.split("/")[-2:]
    benchmark = tokens[0]
    foldover, row, size = tokens[1].split("_")
    actual_row = int(foldover) * int(size) + int(row)
    return actual_row, benchmark


# This function reads the file which contains the pb matrix for the
# parameter setup and returns that matrix in int array format
def read_pbfile(directory, file, size, param_num):
    columns = [[] * size for i in range(0, size)]
    lines = []
    with open(os.path.join(directory, file), "r") as h_file:
        line = h_file.readlines()
        for i in range(0, param_num):
            for j in range(0, size):
                lines.append(int(line[j].split(",")[i].strip()))
            columns[i] = lines
            lines = []
    return columns


# This Function reads the stats from the directory where the results of
# the simulation runs are contained and returns a matrix which contains
# the correspondent row of the pb matrix, the benchmark to which it
# belongs and the execution time it took to complete the simulation
def get_sim_stats(statsdir):
    sim_stats = []
    for subdir, dirs, files in os.walk(statsdir):
        if "stats.txt" in files:
            row, benchmark = parse_path(subdir)
            sim_second = None
            with open(os.path.join(subdir, "stats.txt"), "r") as stats_file:
                sim_second = float(stats_file.readlines()[2].split()[1])
                sim_stats.append([row, benchmark, sim_second])
    return sim_stats


# This function calculates the efficiencies of each parameter given a
# specific benchmark
def calculate_eff(sim_stats, pbcolumns, benchmark, matrix_sz, param_num):
    fd_matrix_sz = matrix_sz * 2

    # Find where in the results matrix the benchmark we are looking for
    # starts and record that index so that we can read it's values later
    bench_start = 0
    bench_num = int(len(sim_stats) / fd_matrix_sz)
    for i in range(0, bench_num):
        if sim_stats[i * fd_matrix_sz][1] == benchmark:
            bench_start = i * fd_matrix_sz

    # Create the PB matrix with a foldover as an np array so that we can
    # perform the dot product on it later
    columns = []
    for i in range(0, fd_matrix_sz):
        if i < matrix_sz:
            foldover = np.array(pbcolumns[i])
            columns.append(foldover)
        else:
            foldover = np.array(pbcolumns[i - matrix_sz])
            foldover *= -1
            columns[i - matrix_sz] = np.append(
                columns[i - matrix_sz], foldover
            )

    # This portion creates a new array for the execution times of each
    # configuration and arranges them in the correct order
    exec_time = []
    for i in range(0, fd_matrix_sz):
        for j in range(0, fd_matrix_sz):
            if sim_stats[bench_start + j][0] == i:
                exec_time.append(sim_stats[bench_start + j][2])
                break
            else:
                continue

    # Perform the dot product between the execution times and the setup
    # of the parameter (either 1 or 0)
    eff = []
    for i in range(0, param_num):
        dot = np.dot(columns[i], exec_time)
        eff.append(dot)

    return eff


def get_results(efficiencies, parameters, parameter_num):
    stats = []
    highest_value = 0
    for i in range(0, parameter_num):
        abs_value = (
            efficiencies[i] if efficiencies[i] >= 0 else efficiencies[i] * -1
        )
        highest_value = (
            abs_value if abs_value > highest_value else highest_value
        )
        stats.append([parameters[i], efficiencies[i], abs_value])

    for i in range(0, parameter_num):
        normalization = stats[i][2] / highest_value
        stats[i].append(normalization)

    return sorted(stats, key=lambda stats: stats[2], reverse=True)


if len(sys.argv) < 6:
    print(
        "Usage: python read_stats.py [statsdir] [pbdir] [pbfile] [benchmark] [matrix_size] [parameter_num]"
    )
    print("\n")
    print(
        "statsdir: The directory where the stats are stored in. See README for required structure"
    )
    print("pbdir: The directory where PB matrix file is stored in")
    print("pbfile: The name of the file")
    print("benchmark: Benchmark to calculate efficiencies for")
    print("matrix_size: The size of the matrix")
    print("\n")
else:
    statsdir = str(sys.argv[1])
    pbdir = str(sys.argv[2])
    pbfile = str(sys.argv[3])
    benchmark = str(sys.argv[4])
    matrix_size = int(sys.argv[5])
    parameters = [
        "threadPolicy",
        "BP",
        "U74IntFU",
        "U74IntMulFU",
        "U74IntDivFU",
        "U74MemReadFU",
        "U74MemWriteFU",
        "l2_size",
        "l2_assoc",
        "l1dcache-response_latency",
        "l2cache-data_latency",
        "iptw_caches-size",
        "dptw_caches-size",
    ]
    parameter_num = len(parameters)

    sim_stats = get_sim_stats(statsdir)
    pbcolumns = read_pbfile(pbdir, pbfile, matrix_size, parameter_num)
    efficiencies = calculate_eff(
        sim_stats, pbcolumns, benchmark, matrix_size, parameter_num
    )

    results = get_results(efficiencies, parameters, parameter_num)

    # data = pd.DataFrame(results, columns=["Parameter", "Efficiency", "Absolute Efficiency", "Normalization"])
    # data.head(64)
    print("Parameter,Efficiency,Absolute_Value,Normalization")
    print(
        "\n".join([",".join([str(cell) for cell in row]) for row in results])
    )

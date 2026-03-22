import numpy as np

# Dataset
data = [10, 20, 30, 40, 50]

def compute_and_print_stats() -> None:
    """Compute and print basic statistics for the global `data` list."""
    mean = np.mean(data)
    variance = np.var(data)
    std_dev = np.std(data)

    print("Mean: ", mean)
    print("Variance: ", variance)
    print("Standard Deviation: ", std_dev)


if __name__ == "__main__":
    compute_and_print_stats()
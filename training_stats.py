import pandas as pd
import sys


def calculate_mean_variance_last_row(data):
    """
    Calculates the mean and variance of the last row across all value columns in a DataFrame.

    Parameters:
    data (DataFrame): The DataFrame containing 'step' and multiple value columns.

    Returns:
    tuple: mean, variance, mean - variance, mean + variance of the last row across value columns.
    """
    last_row = data.iloc[-1, 1:]  # Exclude the first 'Step' column
    mean = last_row.mean()
    variance = last_row.std()
    return mean, variance, mean - variance, mean + variance


def main():
    """
    Main function to process a CSV file specified via command line argument.
    """
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    try:
        data = pd.read_csv(csv_path)
        mean, variance, mean_minus_variance, mean_plus_variance = calculate_mean_variance_last_row(data)
        print(f"Mean: {mean}, Variance: {variance}")
        print(f"Mean - Variance: {mean_minus_variance}, Mean + Variance: {mean_plus_variance}")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

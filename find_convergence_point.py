from utilities.general_utils import analyze_chunk_gradients_with_cost, find_iteration_below_cost, get_final_cost_stats
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find the chunk of near-zero gradient with the lowest average value in a CSV file.')
    parser.add_argument('--main', type=str, help='Path to the CSV file containing the trajectory loss data.')
    parser.add_argument('--target', type=str, help='Path to the CSV file containing the trajectory loss data.')
    args = parser.parse_args()
    result = analyze_chunk_gradients_with_cost(args.main)
    print("Convergence chunks", result)
    result = find_iteration_below_cost(args.main, args.target)
    print("Iteration to lowest", result)
    result = get_final_cost_stats(args.main)
    print("Final Costs stats", result)




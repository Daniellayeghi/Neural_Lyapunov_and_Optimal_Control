import pandas as pd
import zipfile
import os
import sys


def process_and_combine_csv_fixed_steps(zip_path, name):
    """
    Extracts CSV files from a zip file, processes them by removing the 'Wall time' column,
    resetting the 'Step' values to start incrementally from 0 regardless of their original values,
    and stacks the 'Value' columns from different files next to each other.

    Parameters:
    zip_path (str): The path to the zip file containing the CSV files.

    Returns:
    str: Path to the new CSV file with combined and correctly processed data.
    """
    extracted_folder_path = zip_path.replace('.zip', '_extracted')
    if not os.path.exists(extracted_folder_path):

        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder_path)

        combined_data = pd.DataFrame()

        for i, file in enumerate(sorted(os.listdir(extracted_folder_path))):
            if file.endswith('.csv'):
                file_path = os.path.join(extracted_folder_path, file)
                data = pd.read_csv(file_path)

                # Ignore 'Wall time' and directly use an incremental step value
                data = data.assign(Step=range(len(data)))

                # Rename 'Value' column to include the file index for uniqueness
                value_column_name = f'Value_{i + 1}'
                data.rename(columns={'Value': value_column_name}, inplace=True)

                # If it's the first file, initialize combined_data
                if combined_data.empty:
                    combined_data = data[['Step', value_column_name]]
                else:
                    combined_data = pd.merge(combined_data, data[['Step', value_column_name]], on='Step', how='outer')

        # Fill any NaN values resulted from outer join
        combined_data.fillna(method='ffill', inplace=True)

        # Save the combined data to a new CSV file
        output_csv_path = f'{extracted_folder_path}/{name}.csv'
        combined_data.to_csv(output_csv_path, index=False)

        return output_csv_path


def main():
    """
    Main function that takes a zip file path as an argument from the terminal and processes the contained CSV files.
    """
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_zip_file>")
        sys.exit(1)

    zip_path = sys.argv[1]
    name = sys.argv[2]

    # Process the CSV files and combine them
    new_csv_path_fixed = process_and_combine_csv_fixed_steps(zip_path, name)
    print(f"Processed and combined CSV saved to: {new_csv_path_fixed}")


if __name__ == "__main__":
    main()

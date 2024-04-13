import pandas as pd
import os

# Function to split the CSV files
def split_csv_files(input_dir, output_dir):
    # List all CSV files in the input directory
    csv_files = [file for file in os.listdir(input_dir) if file.endswith(".csv")]

    # Define the columns for each subset
    column_sets = [
        ['timestamp', 'close', 'high', 'low', 'predicted_timestamp', 'close_model_0', 'close_0', 'high_0', 'low_0'],
        ['timestamp', 'close', 'high', 'low', 'predicted_timestamp', 'close_model_1', 'close_1', 'high_1', 'low_1'],
        ['timestamp', 'close', 'high', 'low', 'predicted_timestamp', 'close_model_2', 'close_2', 'high_2', 'low_2'],
        ['timestamp', 'close', 'high', 'low', 'predicted_timestamp', 'close_model_3', 'close_3', 'high_3', 'low_3'],
        ['timestamp', 'close', 'high', 'low', 'predicted_timestamp', 'close_model_4', 'close_4', 'high_4', 'low_4'],
        ['timestamp', 'close', 'high', 'low', 'predicted_timestamp', 'close_model_5', 'close_5', 'high_5', 'low_5'],
        ['timestamp', 'close', 'high', 'low', 'predicted_timestamp', 'close_model_6', 'close_6', 'high_6', 'low_6'],
        ['timestamp', 'close', 'high', 'low', 'predicted_timestamp', 'close_model_7', 'close_7', 'high_7', 'low_7']
    ]

    # Loop through each CSV file
    for csv_file in csv_files:
        # Read the original CSV file
        df = pd.read_csv(os.path.join(input_dir, csv_file))

        # Create output files for each column set
        for i, columns in enumerate(column_sets):
            output_file = os.path.join(output_dir, f"{csv_file.split('.')[0]}_{i}.csv")
            df[columns].to_csv(output_file, index=False)

# Specify the input directory containing all CSV files and the output directory
input_directory = "./predictions"
output_directory = "./predictions_set"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Call the function to split the CSV files
split_csv_files(input_directory, output_directory)

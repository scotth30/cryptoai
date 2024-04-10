import os
import pandas as pd
from datetime import datetime, timedelta
import logging
from scipy.interpolate import CubicSpline, interp1d
import numpy as np

# Configure logging
logging.basicConfig(filename='imputation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory containing your CSV files (relative path)
directory = 'binancefetch'
# Output directory for filtered CSV files (relative path)
output_directory = 'select_data'
# Column names
column_names = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades', 'timestamp_unix']

def filter_csv(file_path):
    logging.info(f"Filtering CSV file: {file_path}")
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    pair, interval = file_name.split('_')
    interval = int(interval)
    df = pd.read_csv(file_path)
    df['timestamp_unix'] = df['timestamp']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    start_date = datetime.now() - timedelta(days=365 * 5)
    filtered_df = df[df['timestamp'] >= start_date].sort_values(by='timestamp')

    
    full_range = pd.date_range(start=filtered_df['timestamp'].min(), end=filtered_df['timestamp'].max(), freq=f'{interval}min')
    filtered_df = filtered_df.set_index('timestamp').reindex(full_range).reset_index().rename(columns={'index': 'timestamp'})
    
    # Apply .astype here as well
    filtered_df['timestamp'] = filtered_df['timestamp_unix']
    filtered_df = filtered_df.drop(columns=['timestamp_unix'])

    
    output_file = os.path.join(output_directory, file_name + '.csv')
    filtered_df.to_csv(output_file, index=False)
    logging.info(f"Filtered data saved to CSV file: {output_file}")
    return filtered_df, pair, interval


def perform_time_series_imputation_batch(df_batch):
    logging.info("Performing time-series imputation on batch")
    # Define the window size for the moving average based on your interval or preference
    window_size = 5  # Example: You might adjust this based on your specific needs
    
    for col in ['open', 'high', 'low', 'close', 'volume', 'trades']:
        # Apply moving average imputation
        df_batch[col] = df_batch[col].fillna(df_batch[col].rolling(window=window_size, min_periods=1, center=True).mean())
        
        # Apply linear interpolation for any remaining missing values
        df_batch[col] = df_batch[col].interpolate(method='linear')
        
        # Ensure there are no negative values for any column
        df_batch[col] = df_batch[col].clip(lower=0)
    

    
    return df_batch


def process_file(file_path):
    filtered_df, pair, interval = filter_csv(file_path)
    df_imputed = pd.DataFrame()
    for start in range(0, len(filtered_df), 1000):
        end = min(start + 1000, len(filtered_df))
        df_batch = filtered_df.iloc[start:end].copy()
        df_batch_imputed = perform_time_series_imputation_batch(df_batch)
        df_imputed = pd.concat([df_imputed, df_batch_imputed], ignore_index=True)
    output_file_path = os.path.join(output_directory, f"{pair}_{interval}_minute_data.csv")
    df_imputed.to_csv(output_file_path, index=False)
    logging.info(f"Imputed data saved to CSV file: {output_file_path}")

current_directory = os.getcwd()
input_directory = os.path.join(current_directory, directory)
output_directory = os.path.join(current_directory, output_directory)
os.makedirs(output_directory, exist_ok=True)
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_directory, filename)
        process_file(file_path)
logging.info("Script execution completed.")


import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
import os
import logging
import pyarrow.parquet as pq
import time


mysql_config = {
    'user': 'root',
    'password': '',
    'host': '192.168.1.122',
    'database': 'crypto'
}

# Setup basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory setup
data_dir = './kraken_5year_processed'

os.makedirs(data_dir, exist_ok=True)


def dynamic_schema_reading(file_path: str) -> pd.Index:
    """Read the schema of a Parquet file to get column names."""
    try:
        parquet_file = pq.ParquetFile(file_path)
        schema = parquet_file.schema.to_arrow_schema()
        return pd.Index(schema.names)
    except Exception as e:
        logger.error(f"Error reading schema for {file_path}: {e}")
        return pd.Index([])

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.clip(lower=0)).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()

    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))

    df['RSI'] = RSI.fillna(50)  # Neutral value for RSI when data is insufficient
    return df

def calculate_bollinger_bands(df, period=20):
    middle_BB = df['close'].rolling(window=period).mean()
    std_dev = df['close'].rolling(window=period).std()

    df['upper_BB'] = middle_BB + (std_dev * 2)
    df['lower_BB'] = middle_BB - (std_dev * 2)
    df['middle_BB'] = middle_BB

    # Fill NaN values with 0 (Consider implications)
    df[['upper_BB', 'lower_BB', 'middle_BB']] = df[['upper_BB', 'lower_BB', 'middle_BB']].fillna(0)

    return df

def calculate_stochastic_oscillator(df, period=14):
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()

    df['%K'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df['%D'] = df['%K'].rolling(window=3).mean()

    # Fill NaN values with 0 (Consider implications)
    df[['%K', '%D']] = df[['%K', '%D']].fillna(0)

    return df


def calculate_obv(df):
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).cumsum().fillna(0)
    return df



def calculate_average_true_range(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=period).mean().fillna(0)

    
    return df

def add_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    return df

def calculate_lagged_features(df, feature, lags):
    for lag in lags:
        df[f'{feature}_lag_{lag}'] = df[feature].shift(lag).fillna(0)
    return df

def calculate_volume_pct_change(df):
    try:
        if 'volume' in df.columns:
            volume_change = df['volume'].pct_change() * 100
            volume_change.replace([np.inf, -np.inf], np.nan, inplace=True)
            df['volume_pct_change'] = volume_change.fillna(0)
        else:
            logger.error("Column 'volume' not found in the DataFrame.")
    except Exception as e:
        logger.error(f"Error calculating volume percentage change: {e}")
    return df

def enhanced_feature_engineering(df):
    
    df = df.sort_values(by='timestamp')
    if 'timestamp_unix' in df.columns:
        df = df.drop(columns=['timestamp_unix'])
    df = calculate_rsi(df)
    df = calculate_bollinger_bands(df)
    df = calculate_obv(df)
    df = calculate_stochastic_oscillator(df)
    df = add_time_features(df)
    df = calculate_average_true_range(df)
    df = calculate_volume_pct_change(df)

    volatility_index = df['ATR'].rolling(window=14).mean()
    
    

    dynamic_span_short = np.where(volatility_index > volatility_index.median(), 9, 12)
    dynamic_span_long = np.where(volatility_index > volatility_index.median(), 21, 26)
    df['EMA_Dynamic_Short'] = df['close'].ewm(span=dynamic_span_short.mean(), adjust=False).mean().ffill()
    df['EMA_Dynamic_Long'] = df['close'].ewm(span=dynamic_span_long.mean(), adjust=False).mean().ffill()
    df['EMA_7'] = df['close'].ewm(span=7, adjust=False).mean().ffill()
    df['EMA_17'] = df['close'].ewm(span=17, adjust=False).mean().ffill()
    df['EMA_23'] = df['close'].ewm(span=23, adjust=False).mean().ffill()
    df['EMA_37'] = df['close'].ewm(span=37, adjust=False).mean().ffill()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean().ffill()
    df['EMA_100'] = df['close'].ewm(span=100, adjust=False).mean().ffill()
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean().ffill()
    df['MACD_Dynamic'] = df['EMA_Dynamic_Short'] - df['EMA_Dynamic_Long']
    df['MACD_Signal_Dynamic'] = df['MACD_Dynamic'].ewm(span=9, adjust=False).mean().ffill()
    
    logger.debug("Calculated EMA, MACD")
    # Calculate log returns, ensure no log of zero, and fill initial NA with 0
    df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)

    # Backward fill for the Simple Moving Average over 7 days
    df['SMA_7'] = df['close'].rolling(window=7).mean().bfill()
    df['SMA_17'] = df['close'].rolling(window=14).mean().bfill()
    df['SMA_23'] = df['close'].rolling(window=21).mean().bfill()
    df['SMA_37'] = df['close'].rolling(window=30).mean().bfill()
    df['SMA_50'] = df['close'].rolling(window=50).mean().bfill()
    df['SMA_100'] = df['close'].rolling(window=100).mean().bfill()
    df['SMA_200'] = df['close'].rolling(window=200).mean().bfill()


    # Percentage changes for open, close, high, and low prices without deprecated behavior
    df['open_pct_change'] = df['open'].pct_change(fill_method=None).fillna(0) * 100
    df['close_pct_change'] = df['close'].pct_change(fill_method=None).fillna(0) * 100
    df['high_pct_change'] = df['high'].pct_change(fill_method=None).fillna(0) * 100
    df['low_pct_change'] = df['low'].pct_change(fill_method=None).fillna(0) * 100

    # Forward fill for volume to handle missing values
    df['volume'] = df['volume'].ffill()

    # Define lagged feature configurations and apply them
    lag_feature_configs = [
        {
            'features': ['open_pct_change', 'close_pct_change', 'volume_pct_change','high_pct_change', 'low_pct_change', 'MACD_Dynamic', 'log_return', '%D', 'RSI'],
            'lags': [1, 7, 11, 23, 57, 97]
        }
    ]

    for config in lag_feature_configs:
        for feature in config['features']:
            df = calculate_lagged_features(df, feature, config['lags'])
    return df

# Updated process_file function to ensure proper handling with Dask
def process_file(file_path: str) -> str:
    """Process and normalize a single file using pandas."""
    try:
        logging.info(f"Processing: {file_path}")
        
        # Read parquet file directly into a pandas DataFrame
        df = pd.read_parquet(file_path, engine='pyarrow')
        logging.info(f"Original features: {df.columns.tolist()}")
        
        # Apply enhanced feature engineering on the DataFrame
        df = enhanced_feature_engineering(df)  # Ensure this function is compatible with pandas DataFrame
        logging.info(f"Enhanced features: {df.columns.tolist()}")
        
        # Define the path for the processed file
        processed_file_path = os.path.join(data_dir, os.path.basename(file_path).replace('_imported.parquet', '_processed.parquet'))
        
        # Save the processed DataFrame back to a Parquet file
        df.to_parquet(processed_file_path, engine='pyarrow', index=False)  # index=False to avoid saving DataFrame index as a column
        logger.info(f"Processed and saved: {processed_file_path}")
        
        return processed_file_path
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return ''
def seconds_to_hms(seconds):
    """Convert seconds to hours:minutes:seconds format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h:{int(minutes)}m:{int(seconds)}s"


def main():
    directory = './select_data'
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
    processed_files = []

    for file_path in files:
        logging.info(f"Processing file: {file_path}")
        processed_file_path = process_file(file_path)
        if processed_file_path:
            logger.info(f"Processed file saved to: {processed_file_path}")
            processed_files.append(processed_file_path)



if __name__ == "__main__":
    main()



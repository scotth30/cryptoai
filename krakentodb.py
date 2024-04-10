import dask
dask.config.set(scheduler='threads', num_workers=4)
import dask.dataframe as dd
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar
from sklearn.preprocessing import RobustScaler
import os
import logging
import pyarrow.parquet as pq
import time
import mysql.connector
from mysql.connector import Error
from datacleanernew import enhanced_feature_engineering
import pandas as pd

mysql_config = {
    'user': 'root',
    'password': '',
    'host': '192.168.1.122',
    'database': 'crypto'
}



cryptocurrencies = [
    "BTC", "ETH", "LPT", "WAVES", "SUI", "DOT", "DOGE", "ADA", "ETC", "KAVA", "XRP", "BCH", "ARB", "EOS", "SUSHI", "LINK", "UNI", "APE", "APT"
]
# Setup basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory setup
data_dir = './binancefetch'
os.makedirs(data_dir, exist_ok=True)




def create_table_if_not_exists(cursor, table_name, df_columns):
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            `timestamp` DATETIME NOT NULL PRIMARY KEY,
            {", ".join([f"`{col}` FLOAT" for col in df_columns if col not in ['timestamp']])}
        )
    """)

def ensure_table_creation_and_columns(cursor, table_name, df_columns):
    create_table_if_not_exists(cursor, table_name, df_columns)
    cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
    existing_columns = {column[0] for column in cursor.fetchall()}
    missing_columns = set(df_columns) - existing_columns
    for col in missing_columns:
        cursor.execute(f"ALTER TABLE `{table_name}` ADD COLUMN `{col}` FLOAT")
        
numerical_cols = ['open', 'close', 'high', 'low', 'volume', 'trades']

def process_file(file_path: str):
    logger.info(f"Processing file: {file_path}")
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')


    enhanced_ddf = enhanced_feature_engineering(df)

    logger.info(enhanced_ddf.head(10).to_string())
    enhanced_ddf.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Forward fill NaN values
    enhanced_ddf.ffill(inplace=True)

    # Replace NaN values with a placeholder value, like 0
    enhanced_ddf.fillna(0, inplace=True)
    try:
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()

        # Deriving table name from file name
        symbol = os.path.basename(file_path).split('_')[0].upper()
        table_name = f"{symbol}1mi".lower()

        # Ensure table and columns exist
        ensure_table_creation_and_columns(cursor, table_name, enhanced_ddf.columns)


        # Insert or update records
        for index, row in enhanced_ddf.iterrows():
            placeholders = ', '.join(['%s'] * len(enhanced_ddf.columns))
            columns = ', '.join([f"`{col}`" for col in enhanced_ddf.columns])
            values = tuple(row[col] for col in enhanced_ddf.columns)
            sql = f"INSERT INTO `{table_name}` ({columns}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE " + ', '.join([f"`{col}`=VALUES(`{col}`)" for col in enhanced_ddf.columns])
            cursor.execute(sql, values)

        connection.commit()
    except Error as e:
        logger.error(f"Database error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def main():
    data_dir = './binancefetch'
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    for file_path in files:
        process_file(file_path)

if __name__ == "__main__":
    main()
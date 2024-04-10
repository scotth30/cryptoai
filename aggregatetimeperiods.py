import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error, IntegrityError 
import logging
import sqlalchemy
from sqlalchemy import create_engine
from datacleanernew import enhanced_feature_engineering
from sqlalchemy import text
from sqlalchemy import inspect
# Database Configuration
mysql_config = {
    'user': 'root',
    'password': '',
    'host': '192.168.1.122',
    'database': 'crypto'
}

# SQLAlchemy engine for bulk inserts
engine = create_engine(f"mysql+mysqlconnector://{mysql_config['user']}:{mysql_config['password']}@{mysql_config['host']}/{mysql_config['database']}")

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Aggregation Windows
aggregation_windows = [ '3min', '5min', '10min', '15min', '20min', '30min', '60min', '120min', '720min']


# Custom aggregation function example
def custom_aggregation_function(x):
    weights = np.arange(1, len(x) + 1)  # Example: linear weights
    return (x * weights).sum() / weights.sum()

def fetch_data(table_name, cursor):
    """Fetch data from the database."""
    try:
        query = f"SELECT * FROM {table_name} ORDER BY timestamp ASC"
        cursor.execute(query)
        rows = cursor.fetchall()
        if rows:
            columns = cursor.column_names
            return pd.DataFrame(rows, columns=columns)
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no rows are fetched
    except Error as e:
        da.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

def aggregate_data(df, freq):
    # Set 'timestamp' column as the index
    aggregated_df = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'trades': 'sum',
    }).reset_index()
    
    updated_df = enhanced_feature_engineering(aggregated_df)  # Apply custom feature engineering
    return updated_df

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
def save_to_database(df, table_name,cursor):
    """
    Ensure table creation with an index on 'timestamp', then save the DataFrame.
    """
    ensure_table_creation_and_columns(cursor, table_name, df.columns.tolist())

    # Make sure 'timestamp' is the DataFrame's index and not a column
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)  # Sort the DataFrame by the index
        df = df.iloc[200:]
    try:
        with engine.connect() as conn:
            transaction = conn.begin()
            
            # Delete rows with duplicate timestamps
            for ts in df.index:
                query = f"DELETE FROM {table_name} WHERE timestamp = '{ts}'"
                conn.execute(text(query))
            
            df.to_sql(name=table_name, con=conn, if_exists='append', index=True, index_label='timestamp', method='multi', chunksize=25000)
            transaction.commit()
            logger.info(f"Data saved to table: {table_name}")
    except Exception as e:
        logger.error(f"Error saving data to table {table_name}: {e}")
        transaction.rollback()
    finally:
        conn.close()

    logger.info(f"Data saved to table: {table_name}")



def process_cryptocurrency(symbol, df, cursor):
    """
    Process and save aggregated data for a single cryptocurrency.
    """
    for window in aggregation_windows:
        aggregated_df = aggregate_data(df, window)
        save_to_database(aggregated_df, f"{symbol.lower()}{window[:-1]}", cursor)

def main():
    cryptocurrencies = ["BTC", "ETH", "LPT", "WAVES", "SUI", "DOT",  "ADA", "ETC", "KAVA", "XRP", "BCH", "ARB", "EOS", "SUSHI", "LINK", "UNI", "APE"]
    connection = mysql.connector.connect(**mysql_config)
    cursor = connection.cursor()

    for symbol in cryptocurrencies:
        logger.info(f"Processing {symbol}")
        df = fetch_data(f"{symbol}1mi", cursor)
        if df.empty:  # Now this check is safe, as df is guaranteed to be a DataFrame
            logger.info(f"No data found for {symbol}, skipping...")
            continue
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        process_cryptocurrency(symbol, df, cursor)
    
    cursor.close()
    connection.close()

if __name__ == "__main__":
    main()

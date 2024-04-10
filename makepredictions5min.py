import pandas as pd
from sqlalchemy import create_engine
import joblib
import os
from datetime import datetime, timedelta
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
mysql_config = {
    'user': 'root',
    'password': '',
    'host': '192.168.1.122',
    'database': 'crypto'
}
engine = create_engine(f"mysql+mysqlconnector://{mysql_config['user']}:{mysql_config['password']}@{mysql_config['host']}/{mysql_config['database']}")

# Models directory
close_dir = './close'
high_dir = './high'
low_dir = './low'
predictions_dir = './predictions'

def fetch_most_recent_row(crypto_symbol):
    table_name = f"{crypto_symbol.lower()}5mi"
    query = f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT 1"
    df = pd.read_sql(query, engine)
    return df

def load_models_for_type(model_type):
    if model_type == "close":
        models_path = close_dir
    elif model_type == "high":
        models_path = high_dir
    elif model_type == "low":
        models_path = low_dir
    else:
        logger.error(f"Invalid model type: {model_type}")
        return {}

    model_files = os.listdir(models_path)
    models = {}
    for model_file in model_files:
        if model_file.endswith(".joblib"):
            # Extract cryptocurrency symbol from the filename
            symbol = ''.join(filter(str.isupper, model_file.split('_')[0]))
            model_path = os.path.join(models_path, model_file)
            if symbol not in models:
                models[symbol] = []
            models[symbol].append((model_type, joblib.load(model_path)))  # Storing model type along with model
    return models

def predict_and_save(crypto_symbols, model_types):
    for symbol in crypto_symbols:
        symbol_data = []  # To store data for each symbol
        for model_type in model_types:
            models = load_models_for_type(model_type)
            if not models or symbol not in models:
                logger.error(f"No models found for {model_type} for {symbol}")
                continue

            for model_info in models[symbol]:
                model_type_used, model = model_info
                try:
                    df = fetch_most_recent_row(symbol)

                    if df.empty:
                        logger.error(f"No data found for {symbol}")
                        continue

                    features = df.drop(['timestamp', model_type, f'predicted_{model_type}'], axis=1, errors='ignore')
                    predicted_value = model.predict(features)[0]

                    timestamp = df['timestamp'].iloc[0]
                    prediction_timestamp = timestamp + timedelta(minutes=5)  # Adjust the timedelta as needed

                    # Calculate percentage difference
                    price = df[model_type].iloc[0]
                    percent_diff = ((predicted_value - price) / price) * 100

                    # Add data to the symbol's list
                    symbol_data.append({
                        'timestamp': timestamp,
                        'price': price,
                        'predicted_timestamp': prediction_timestamp,
                        'predicted_price': predicted_value,
                        'percent_diff': percent_diff,
                        'model_type_used': model_type_used
                    })

                except Exception as e:
                    logger.error(f"Error processing {symbol} ({model_type}): {e}")

        if not symbol_data:
            continue

        # Create DataFrame for the symbol's data
        symbol_df = pd.DataFrame(symbol_data)

        # File path to save CSV
        csv_file_path = os.path.join(predictions_dir, f"{symbol}_5min_predictions.csv")
        # Append to existing CSV or create a new one
        if os.path.exists(csv_file_path):
            symbol_df.to_csv(csv_file_path, mode='a', header=False, index=False)
        else:
            symbol_df.to_csv(csv_file_path, index=False)

if __name__ == "__main__":
    crypto_symbols = ["ETH", "BTC", "LPT", "WAVES", "SUI", "DOT", "ADA", "ETC", "KAVA", "XRP", "BCH", "EOS", "SUSHI", "LINK", "UNI", "APE", "APT"]

    model_types = ["close", "high", "low"]

    predict_and_save(crypto_symbols, model_types)
    time.sleep(60)  # Wait for one minute before moving to the next model type

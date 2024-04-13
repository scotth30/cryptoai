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

def fetch_most_recent_rows(crypto_symbol, days=7):
    table_name = f"{crypto_symbol.lower()}5mi"
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    query = f"SELECT * FROM {table_name} WHERE timestamp BETWEEN '{start_time}' AND '{end_time}' ORDER BY timestamp ASC"
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
            model = joblib.load(model_path)
            if symbol not in models:
                models[symbol] = []
            models[symbol].append((model_type, model, model_file))  # Store model file name along with model
    return models

def generate_output_line(timestamp, symbol_data, actual_close, actual_high, actual_low):
    output_line = {
        'timestamp': timestamp,
        'close': actual_close,
        'high': actual_high,
        'low': actual_low,
        'predicted_timestamp': timestamp + timedelta(minutes=5)
    }

    close_models = []
    high_models = []
    low_models = []

    for model_data in symbol_data:
        model_file_name = model_data['model_file_name']
        model_file_parts = model_file_name.split('_')
        symbol = ''.join(filter(str.isupper, model_file_parts[0]))
        time_period = model_file_parts[1] if len(model_file_parts) > 1 else None
        trained_objective = model_file_parts[2] if len(model_file_parts) > 2 else None
        objective = None
        if len(model_file_parts) > 3:
            objective_part = model_file_parts[3].split('.')[0]
            if objective_part:
                objective = objective_part

        model_type_used = model_data['model_type_used']
        predicted_price = model_data['predicted_price']
        percent_diff = model_data['percent_diff']

        if model_type_used == 'close':
            close_models.append((model_file_name, predicted_price, percent_diff, objective, trained_objective))
        elif model_type_used == 'high':
            high_models.append((model_file_name, predicted_price, percent_diff, objective, trained_objective))
        elif model_type_used == 'low':
            low_models.append((model_file_name, predicted_price, percent_diff, objective, trained_objective))

    for i, (close_model_file, close_price, close_percent_diff, close_objective, close_trained_objective) in enumerate(close_models):
        output_line[f'close_model_{i}'] = close_model_file
        output_line[f'close_{i}'] = close_price
        if close_objective and close_trained_objective:
            output_line[f'close_percent_diff_{close_objective}_{close_trained_objective}_{i}'] = close_percent_diff

    for i, (high_model_file, high_price, high_percent_diff, high_objective, high_trained_objective) in enumerate(high_models):
        output_line[f'high_model_{i}'] = high_model_file
        output_line[f'high_{i}'] = high_price
        if high_objective and high_trained_objective:
            output_line[f'high_percent_diff_{high_objective}_{high_trained_objective}_{i}'] = high_percent_diff

    for i, (low_model_file, low_price, low_percent_diff, low_objective, low_trained_objective) in enumerate(low_models):
        output_line[f'low_model_{i}'] = low_model_file
        output_line[f'low_{i}'] = low_price
        if low_objective and low_trained_objective:
            output_line[f'low_percent_diff_{low_objective}_{low_trained_objective}_{i}'] = low_percent_diff

    return output_line

def predict_and_save(crypto_symbols, model_types, days=7):
    for symbol in crypto_symbols:
        symbol_data = {}  # To store data for each timestamp
        for model_type in model_types:
            models = load_models_for_type(model_type)
            if not models or symbol not in models:
                logger.error(f"No models found for {model_type} for {symbol}")
                continue
            for model_info in models[symbol]:
                model_type_used, model, model_file_name = model_info
                try:
                    df = fetch_most_recent_rows(symbol, days=days)
                    if df.empty:
                        logger.error(f"No data found for {symbol}")
                        continue

                    for _, row in df.iterrows():
                        # Create a copy of the row to avoid modifying the original
                        features = row.drop(['timestamp', model_type, f'predicted_{model_type}'], errors='ignore').copy()
                        if model_type_used == 'close' and 'close' in features:
                            # Drop 'close' column for close model
                            features = features.drop('close')
                        elif model_type_used == 'high' and 'high' in features:
                            # Drop 'high' column for high model
                            features = features.drop('high')
                        elif model_type_used == 'low' and 'low' in features:
                            # Drop 'low' column for low model
                            features = features.drop('low')

                        predicted_value = model.predict([features])[0]
                        timestamp = row['timestamp']
                        price = row[model_type]
                        percent_diff = ((predicted_value - price) / price) * 100
                        if timestamp not in symbol_data:
                            symbol_data[timestamp] = []
                        symbol_data[timestamp].append({
                            'price': price,
                            'predicted_price': predicted_value,
                            'percent_diff': percent_diff,
                            'model_type_used': model_type_used,
                            'model_file_name': model_file_name,
                            'actual_close': row['close'],  # Add actual close value
                            'actual_high': row['high'],    # Add actual high value
                            'actual_low': row['low']       # Add actual low value
                        })
                except Exception as e:
                    logger.error(f"Error processing {symbol} ({model_type}): {e}")
        if not symbol_data:
            continue
        # Create output lines for each timestamp
        output_lines = []
        for timestamp, data in symbol_data.items():
            output_line = generate_output_line(timestamp, data, data[0]['actual_close'], data[0]['actual_high'], data[0]['actual_low'])
            output_lines.append(output_line)
        # Create DataFrame for the symbol's data
        symbol_df = pd.DataFrame(output_lines)
        # File path to save CSV
        csv_file_path = os.path.join(predictions_dir, f"{symbol}_5min_7days_predictions.csv")
        # Append to existing CSV or create a new one
        if os.path.exists(csv_file_path):
            symbol_df.to_csv(csv_file_path, mode='a', header=False, index=False)
        else:
            symbol_df.to_csv(csv_file_path, index=False)

if __name__ == "__main__":
    crypto_symbols = ["ETH", "BTC", "LPT", "WAVES", "SUI", "DOT", "ADA", "ETC", "KAVA", "XRP", "BCH", "EOS", "SUSHI", "LINK", "UNI", "APE"]
    model_types = ["close", "high", "low"]
    predict_and_save(crypto_symbols, model_types, days=7)
    time.sleep(60)  # Wait for one minute before moving to the next model type

import pandas as pd
from lightgbm import LGBMRegressor
import logging
import joblib
import json
import optuna
from sqlalchemy import text, Table, MetaData, Column, Float
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy import create_engine, inspect
from lightgbm import early_stopping
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, HalvingGridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error
import numpy as np
import datetime
import time
import os
from sklearn.preprocessing import StandardScaler
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mysql_config = {
    'user': 'root',
    'password': '',
    'host': '192.168.1.122',
    'database': 'crypto'
}

engine = create_engine(f"mysql+mysqlconnector://{mysql_config['user']}:{mysql_config['password']}@{mysql_config['host']}/{mysql_config['database']}")

# Output directory for models and metadata
output_dir = './model_outputs'
models_dir = './saved_models'  # Directory to save models
os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)  # Ensure the directory for models exists

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types in JSON."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return json.JSONEncoder.default(self, obj)

# Adjust TimeSeriesSplit based on detailed analysis of your data
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)

cryptocurrencies = [ "ETH", "BCH", "WAVES", "SUI", "DOT","XRP",  "ADA","KAVA", "ETC", "ARB", "EOS", "SUSHI", "LINK", "UNI", "APE", "LPT", "BTC"]

# Assuming the rest of your setup code is here
def fetch_data(crypto_symbol):
    table_name = f"{crypto_symbol.lower()}5mi"
    #two_years_ago = datetime.datetime.now() - datetime.timedelta(days=2*365)
    # Format the date in a way that's compatible with your SQL database
    #formatted_date = two_years_ago.strftime('%Y-%m-%d %H:%M:%S')
    # Modify the query to fetch data from the last two years
    query = f"SELECT * FROM {table_name} WHERE timestamp ORDER BY timestamp ASC"
    df = pd.read_sql(query, engine)
    return df


def ensure_predicted_low_column(engine, table_name):
    metadata = MetaData()
    try:
        table = Table(table_name, metadata, autoload_with=engine)
    except NoSuchTableError:
        logger.error(f"Table {table_name} does not exist.")
        return False
    
    # Check if column exists
    if 'predicted_low' not in table.c:
        # Correctly wrap the ALTER TABLE statement in text() for execution
        alter_stmt = text(f"ALTER TABLE {table_name} ADD COLUMN predicted_low FLOAT;")
        with engine.begin() as conn:  # Use a transaction for DDL statements
            conn.execute(alter_stmt)
        logger.info(f"Column 'predicted_low' added to {table_name}.")
        return True
    return False

def save_predictions_to_db(predictions, crypto_symbol, timestamps):
    table_name = f"{crypto_symbol.lower()}5mi"  # Assuming predictions are saved back to the main table
    # Ensure the predicted_low column exists
    if not ensure_predicted_low_column(engine, table_name):
        return
    with engine.begin() as conn:  # Using a transaction for safety
        for timestamp, prediction in zip(timestamps, predictions):
            # Parameterized query for safer execution
            stmt = text(f"""
                UPDATE {table_name}
                SET predicted_low = :prediction
                WHERE timestamp = :timestamp
            """)
            conn.execute(stmt, {'prediction': prediction, 'timestamp': timestamp})
    logger.info(f"Predictions saved to database for {crypto_symbol}.")


def save_metadata(best_model, X, y, crypto_symbol):
    predictions = best_model.predict(X)
    metadata = {
        "Best Parameters": best_model.get_params(),
        "MAE": mean_absolute_error(y, predictions),
        "MSE": mean_squared_error(y, predictions),
        "RMSE": np.sqrt(mean_squared_error(y, predictions)),
        "R2": r2_score(y, predictions),
        "MAPE": mean_absolute_percentage_error(y, predictions),
        "Feature Importances": dict(zip(X.columns, best_model.feature_importances_)),
        "Timestamp": datetime.datetime.now().isoformat()
    }
    metadata_path = os.path.join(output_dir, f"{crypto_symbol}5mi_rmse3_low_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4, cls=NumpyEncoder)
    logger.info(f"Metadata saved for {crypto_symbol}.")
    model_filename = os.path.join(models_dir, f"{crypto_symbol}5mi_rmse3_low_model.joblib")
    joblib.dump(best_model, model_filename)
    logger.info(f"Model saved for {crypto_symbol} at {model_filename}")

def preprocess_data(df, crypto_symbol):
    # Fill missing values with median
    for column in df.columns:
        df[column].fillna(df[column].median(), inplace=True)
    return df

    
    return df
def process_crypto(crypto_symbol):
    study_name = "5min_rmse3_low" + crypto_symbol
    try:
        optuna.delete_study(study_name=study_name, storage="sqlite:///db.sqlite3")
    except KeyError:
        logger.info(f"Study {study_name} does not exist in the storage.")
    
    df = fetch_data(crypto_symbol)
    df = preprocess_data(df, crypto_symbol)
    original_timestamps = df['timestamp'].shift(-1)[:-1]
    df['future_low'] = df['low'].shift(-1)
    df = df[:-1]
    X = df.drop(['timestamp', 'low', 'future_low', 'predicted_low'], axis=1, errors='ignore')
    y = df['future_low']

    def objective(trial):

        param = {
            'num_leaves': trial.suggest_int('num_leaves', 10, 2048),
            'max_depth': trial.suggest_int('max_depth', 5, 11),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.13),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 150),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'random_state': 73,
            'boosting_type': 'goss',  # Changed to GOSS boosting
            'n_jobs': -1,
            'device': 'cpu',
            'objective': 'rmse',
            # GOSS-specific parameters
            'top_rate': trial.suggest_float('top_rate', 0.0, 0.5),
            'other_rate': trial.suggest_float('other_rate', 0.0, 0.5),
        }

        # Ensure the sum of top_rate and other_rate is less than 1
        if param['top_rate'] + param['other_rate'] >= 1:
            param['other_rate'] = 1 - param['top_rate']

        print(param)
        try:
            model = LGBMRegressor(**param)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=73)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            preds = model.predict(X_val, num_iteration=model.best_iteration_)

            # Use rmse3 for evaluation
            score = mean_absolute_percentage_error(y_val, preds)
        except ValueError as e:
            print(f"Encountered an error during training: {e}")
            score = 1.0  # Assign a high score to indicate failure
        return score

    study = optuna.create_study(study_name=study_name, direction='minimize', storage="sqlite:///db.sqlite3", load_if_exists=True)
    max_retries = 5
    retries = 0
    while retries < max_retries:
        try:
            study.optimize(objective, n_trials=50)
            break  # Success, exit the loop
        except optuna.exceptions.StorageInternalError as e:
            retries += 1
            logger.error(f"Database locked. Retrying in 5 seconds... (Attempt {retries}/{max_retries})")
            time.sleep(5)
    
    best_params = study.best_params
    print(best_params)
    best_model = LGBMRegressor(**best_params)
    best_model.fit(X, y)

    save_metadata(best_model, X, y, crypto_symbol)
  
    #save_predictions_to_db(best_model.predict(X), crypto_symbol, original_timestamps)



def main():
    for crypto_symbol in cryptocurrencies:
        logger.info(f"Processing {crypto_symbol}")
        process_crypto(crypto_symbol)

if __name__ == "__main__":
    main()

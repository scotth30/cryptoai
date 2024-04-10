import pandas as pd
import logging
import json
import optuna
from sqlalchemy import create_engine
import numpy as np
import datetime
import os
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mysql_config = {
    'user': 'root',
    'password': '',
    'host': '192.168.1.122',
    'database': 'crypto'
}

engine = create_engine(f"mysql+mysqlconnector://{mysql_config['user']}:{mysql_config['password']}@{mysql_config['host']}/{mysql_config['database']}")

output_dir = './model_outputs'
models_dir = './saved_models'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

cryptocurrencies = ["ETH", "BCH", "WAVES", "SUI", "DOT", "XRP", "ADA", "KAVA", "ETC", "ARB", "EOS", "SUSHI", "LINK", "UNI", "APE", "APT", "LPT", "BTC"]

def fetch_data(crypto_symbol):
    table_name = f"{crypto_symbol.lower()}1mi"
    query = f"SELECT * FROM {table_name} WHERE timestamp ORDER BY timestamp ASC"
    df = pd.read_sql(query, engine)
    return df

def preprocess_data(df, crypto_symbol):
    for column in df.columns:
        df[column].fillna(df[column].median(), inplace=True)
    return df

def build_model(input_dim, params):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mean_absolute_percentage_error')
    return model

def train_model(X_train, y_train, X_val, y_val, params, crypto_symbol):
    model_filename = os.path.join(models_dir, f"{crypto_symbol}_keras_model.h5")
    model = build_model(X_train.shape[1], params)
    early_stopping = EarlyStopping(patience=15, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=params['n_epochs'], batch_size=params['batch_size'], callbacks=[early_stopping])
    model.save(model_filename)
    return model

def process_crypto(crypto_symbol):
    df = fetch_data(crypto_symbol)
    df = preprocess_data(df, crypto_symbol)
    df['future_close'] = df['close'].shift(-1)
    df.drop(['timestamp', 'close'], axis=1, inplace=True)
    df.dropna(inplace=True)
    X = df.drop('future_close', axis=1).values
    y = df['future_close'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    def objective(trial):
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'n_epochs': trial.suggest_int('n_epochs', 50, 200),
        }
        model = train_model(X_train, y_train, X_val, y_val, params, crypto_symbol)
        predictions = model.predict(X_val)
        mae = mean_absolute_error(y_val, predictions)
        return mae

    study_name = "1minkeras" + crypto_symbol
    study = optuna.create_study(study_name=study_name, direction='minimize')
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    best_model = train_model(X_train, y_train, X_val, y_val, best_params, crypto_symbol)

    model_filename = os.path.join(models_dir, f"{crypto_symbol}_keras_model_final.h5")
    best_model.save(model_filename)
    logger.info(f"Model saved for {crypto_symbol} at {model_filename}")

def main():
    for crypto_symbol in cryptocurrencies:
        logger.info(f"Processing {crypto_symbol}")
        process_crypto(crypto_symbol)

if __name__ == "__main__":
    main()

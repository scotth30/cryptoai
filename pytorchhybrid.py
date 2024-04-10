import pandas as pd
import logging
import json
import optuna
from sqlalchemy import create_engine
import numpy as np
import datetime
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error
from torchmetrics import MeanAbsolutePercentageError
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import joblib


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return json.JSONEncoder.default(self, obj)

cryptocurrencies = ["ETH", "BCH", "WAVES", "SUI", "DOT", "XRP", "ADA", "KAVA", "ETC", "ARB", "EOS", "SUSHI", "LINK", "UNI", "APE", "APT", "LPT", "BTC"]

def fetch_data(crypto_symbol):
    table_name = f"{crypto_symbol.lower()}1mi"
    query = f"SELECT * FROM {table_name} WHERE timestamp ORDER BY timestamp ASC"
    df = pd.read_sql(query, engine)
    return df

def preprocess_data(df, crypto_symbol):
    for column in df.columns:
        # Avoid chained assignment by directly updating the column
        df[column] = df[column].fillna(df[column].median())
    return df


class HybridNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, dropout_rate=0.2):
        super(HybridNeuralNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        
        layers = []
        prev_dim = hidden_dim
        for out_dim in [512, 256, 128, 64, 32, 16, 8, 4, 2]:
            layers.extend([
                nn.Linear(prev_dim, out_dim),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU()
            ])
            prev_dim = out_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.ReLU())  # Adding ReLU activation for regression task

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Ensure x is 3-dimensional (batch_size, seq_len, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a sequence length of 1
        batch_size, seq_len, _ = x.size()
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.contiguous().view(batch_size, -1)
        x = self.model(lstm_out)
        return x

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.val_loss_min = np.Inf
        self.optimizer = None  # Initialize optimizer here

    def __call__(self, val_loss, model, model_path, optimizer):
        self.optimizer = optimizer  # Set the optimizer
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_path):
        if val_loss < self.val_loss_min:
            logger.info(f'Validation loss decreased ({self.val_loss_min} --> {val_loss}). Saving model ...')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_path)
            self.val_loss_min = val_loss



def train_model(X_train, y_train, X_val, y_val, params, crypto_symbol):
    model_filename = os.path.join(models_dir, f"{crypto_symbol}_pytorch_model.pth")
    input_size = X_train.shape[1]  # Assuming X_train is (batch_size, seq_length, input_size)
    
    # Correctly access hidden_dim and num_layers from params
    hidden_dim = params['hidden_dim']
    num_layers = params['num_layers']

    # Instantiate the model with the correct hidden_dim and num_layers
    model = HybridNeuralNetwork(input_size, hidden_dim, num_layers).to(device)
    criterion = nn.SmoothL1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)  # Accessing .values from the Series
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(device) 

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)

    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=params['max_lr'],  # Increase the max_lr by 20%
        epochs=params['n_epochs'],
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # Extend the ramp-up phase
        div_factor=10,  # Increase the initial learning rate divisor
        final_div_factor=1e4,  # Define the final learning rate divisor
        three_phase=True,  # Enable the three-phase schedule
    )
    early_stopping = EarlyStopping(patience=10, delta=0.001)

    # Training loop
    for epoch in range(params['n_epochs']):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Log average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        logger.info(f'Epoch [{epoch+1}/{params["n_epochs"]}], Loss: {epoch_loss:.4f}')

        scheduler.step()

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor).squeeze()
            val_loss = criterion(val_predictions, y_val_tensor)
            logger.info(f'Epoch {epoch+1}, Val Loss: {val_loss.item()}, LR: {optimizer.param_groups[0]["lr"]}')
            mape = mean_absolute_percentage_error(y_val_tensor.cpu().numpy(), val_predictions.cpu().numpy())
            logger.info(f'MAPE: {mape}')

        early_stopping(val_loss, model, model_filename, optimizer)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

    # Note: No need to save or load the model state dict here if using Optuna
    return val_loss.item()


def process_crypto(crypto_symbol):
    df = fetch_data(crypto_symbol)
    df = preprocess_data(df, crypto_symbol)
    
    study_name = "1minpytorch" + crypto_symbol
    db_file_path = "db.sqlite3"  # Specify the SQLite database file name
    if os.path.exists(db_file_path):  # Check if the file exists
        try:
            optuna.delete_study(study_name=study_name, storage=f"sqlite:///{db_file_path}")
        except KeyError:
            logger.info(f"Study {study_name} does not exist in the storage.")

    df['future_close'] = df['close'].shift(-1)
    columns_to_drop = ['timestamp', 'close', 'predicted_close']
    df = df.drop(columns=columns_to_drop, errors='ignore').dropna()

    X = df.drop('future_close', axis=1)
    y = df['future_close']

    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f'{models_dir}/{crypto_symbol}_pytorch_scaler.pkl')  # Saving the scaler

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 0.005, 0.007)
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
        max_lr = trial.suggest_float('max_lr', learning_rate, learning_rate * 1.1)
        n_epochs = trial.suggest_int('n_epochs', 100, 200)

        # Correctly access hidden_dim and num_layers
        hidden_dim = trial.suggest_int('hidden_dim', 64, 128, log=True)  # Log scale for hidden_dim
        num_layers = trial.suggest_int('num_layers', 7, 10)

        params = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'max_lr': max_lr,
            'n_epochs': n_epochs,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers
        }

        logger.info(f"Suggested Parameters for Trial: {trial.number}")
        logger.info(json.dumps(params, indent=4))

        val_loss = train_model(X_train, y_train, X_val, y_val, params, crypto_symbol)
        return val_loss



    study = optuna.create_study(study_name=study_name, direction='minimize', storage=f"sqlite:///{db_file_path}", load_if_exists=True)
    study.optimize(objective, n_trials=5)



def save_metadata(model, X, y, crypto_symbol, best_params):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
    
    metadata = {
        "Best Parameters": best_params,
        "MAE": mean_absolute_error(y, predictions),
        "MSE": mean_squared_error(y, predictions),
        "RMSE": np.sqrt(mean_squared_error(y, predictions)),
        "R2": r2_score(y, predictions),
        "MAPE": mean_absolute_percentage_error(y, predictions),
        "MSLE": mean_squared_log_error(y, predictions),
        "Timestamp": datetime.datetime.now().isoformat()
    }
    metadata_path = os.path.join(output_dir, f"{crypto_symbol}pytorch_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4, cls=NumpyEncoder)
    logger.info(f"Metadata saved for {crypto_symbol}.")


def main():
    for crypto_symbol in cryptocurrencies:
        logger.info(f"Processing {crypto_symbol}")
        process_crypto(crypto_symbol)

if __name__ == "__main__":
    main()
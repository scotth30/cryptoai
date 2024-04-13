import os
import pandas as pd
import numpy as np
import pnltradealgorithm

cryptocurrencies = ["BTC", "ETH", "LPT", "WAVES", "SUI", "DOT", "ADA", "ETC", "KAVA", "XRP", "BCH", "EOS", "SUSHI", "LINK", "UNI", "APE"]


def use_pnl_algorithm():
    #calculate pnl using pnltradealgorithm
    pass
def evaluate_models_for_all_cryptos_and_save():
    """
    Evaluate the predicted model values against the actual next close, high, and low prices for all cryptocurrencies.
    Calculate various loss functions and save the evaluation results for each cryptocurrency into separate CSV files.

    Returns:
    - results_dict: Dictionary containing evaluation results for each cryptocurrency
    """
    results_dict = {}

    # Get the path to the current script's directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    predictions_folder = os.path.join(script_dir, "predictions")

    for crypto in cryptocurrencies:
        csv_file = os.path.join(predictions_folder, f"{crypto}_5min_predictions.csv")

        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Initialize lists to store results
        model_names = []
        mae_list = []
        mse_list = []
        rmse_list = []
        mape_list = []

        # Iterate over each model for close, high, and low
        for i in range(8):
            close_model_name = df[f"close_model_{i}"].iloc[0]  # Getting the actual model name from the first row
            high_model_name = df[f"high_model_{i}"].iloc[0]
            low_model_name = df[f"low_model_{i}"].iloc[0]

            # Get the predicted close, high, and low prices for this model
            close_pred = df[f"close_{i}"]
            high_pred = df[f"high_{i}"]
            low_pred = df[f"low_{i}"]

            # Get the actual close, high, and low prices
            actual_close = df["close"]
            actual_high = df["high"]
            actual_low = df["low"]

            # Calculate MAE, MSE, RMSE, MAPE for close
            close_mae = np.mean(np.abs(actual_close - close_pred))
            close_mse = np.mean(np.square(actual_close - close_pred))
            close_rmse = np.sqrt(np.mean(np.square(actual_close - close_pred)))
            close_mape = np.mean(np.abs((actual_close - close_pred) / actual_close))

            # Calculate MAE, MSE, RMSE, MAPE for high
            high_mae = np.mean(np.abs(actual_high - high_pred))
            high_mse = np.mean(np.square(actual_high - high_pred))
            high_rmse = np.sqrt(np.mean(np.square(actual_high - high_pred)))
            high_mape = np.mean(np.abs((actual_high - high_pred) / actual_high))

            # Calculate MAE, MSE, RMSE, MAPE for low
            low_mae = np.mean(np.abs(actual_low - low_pred))
            low_mse = np.mean(np.square(actual_low - low_pred))
            low_rmse = np.sqrt(np.mean(np.square(actual_low - low_pred)))
            low_mape = np.mean(np.abs((actual_low - low_pred) / actual_low))

            # Append the values to the lists for close, high, and low
            model_names.append(f"{close_model_name}_close_{i},{high_model_name}_high_{i},{low_model_name}_low_{i}")
            mae_list.append([close_mae, high_mae, low_mae])
            mse_list.append([close_mse, high_mse, low_mse])
            rmse_list.append([close_rmse, high_rmse, low_rmse])
            mape_list.append([close_mape, high_mape, low_mape])

        # Create a DataFrame for this cryptocurrency
        result_df = pd.DataFrame({
            "Model_CLOSE, MODEL_HIGH, MODEL_LOW": model_names,
            "MAE": mae_list,
            "MSE": mse_list,
            "RMSE": rmse_list,
            "MAPE": mape_list
        })

        # Save the result DataFrame to a CSV file for the cryptocurrency
        result_file = os.path.join(script_dir, f"{crypto}_evaluation_results.csv")
        result_df.to_csv(result_file, index=False)

        # Store the result DataFrame in the dictionary
        results_dict[crypto] = result_df

    # Return the dictionary of results
    return results_dict

# Test the function
results = evaluate_models_for_all_cryptos_and_save()

# Print the first few rows of results for each cryptocurrency
for crypto, result_df in results.items():
    print(f"\nCrypto: {crypto}")
    print(result_df)

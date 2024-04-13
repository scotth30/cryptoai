import pandas as pd
import os

starting_balance = 100  # Starting balance in USD, used as initial amount for the trading simulation.
cryptocurrencies = ["BTC", "ETH", "LPT", "WAVES", "SUI", "DOT", "ADA", "ETC", "KAVA", "XRP", "BCH", "EOS", "SUSHI", "LINK", "UNI", "APE"]  # List of cryptocurrencies that may be involved in the trading simulation.

# Function to calculate trades, total P&L, and end balance for a single set of trading predictions
def calculate_trades(balance, df):
    # Extract close, high, and low prices of the current period from dataframe for comparison with predictions.
    current_close = df['close']
    current_high = df['high']
    current_low = df['low']
    
    total_pnl = 0  # Variable to keep track of the running total profit and loss (P&L).

    # Iterate over defined number of prediction models (here, implied to be 8).
    for i in range(8):
        close_col = f'close_{i}'
        high_col = f'high_{i}'
        low_col = f'low_{i}'

        # Check if the necessary columns exist in the dataframe for the current model.
        if close_col in df.columns and high_col in df.columns and low_col in df.columns:
            predicted_close = df[close_col]
            predicted_high = df[high_col]
            predicted_low = df[low_col]
            
            # Iterate over each row in the dataframe to process trades based on predictions.
            for j in range(len(df) - 1):
                # If the predicted close is higher than the current close, a potential trade is considered.
                if predicted_close[j] > current_close[j]:
                    predicted_range = predicted_high[j] - predicted_low[j]
                    buypoint = predicted_low[j] + (predicted_range * 0.05)  # Set buy point at 5% above the predicted low.
                    sellpoint = predicted_high[j] - (predicted_range * 0.05)  # Set sell point at 5% below the predicted high.

                    # Check if the index for the next period is within the dataframe bounds.
                    if j + 1 < len(df):
                        # If predicted conditions are met during the next period, perform trade calculations.
                        if buypoint >= current_low[j+1]:
                            # Shift to get the next period's close, high, and low for calculations.
                            next_close = current_close[j+1]
                            next_high = current_high[j+1]
                            next_low = current_low[j+1]
                            
                            # Calculate net P&L and updated balance after the trade using a custom function.
                            pnl, balance = calculate_net_pnl(balance, buypoint, sellpoint, current_close[j], next_high, next_close, next_low)
                            total_pnl += pnl  # Update running total P&L.

    return total_pnl, balance  # Return the total P&L and end balance after all trades.

# Function to calculate net P&L and balance for each trade based on given parameters.
def calculate_net_pnl(balance, buypoint, sellpoint, current_close, next_high, next_close, next_low):
    amount_to_trade = balance * 0.01  # Decide the trade amount, which is 1% of the current balance.
    amount_purchased = amount_to_trade / buypoint  # Calculate the amount of cryptocurrency purchased.
    kraken_buy_fee = amount_to_trade * 0.0016  # Assume a buy fee from the Kraken exchange.

    # Check if the crypto asset price hit the sell point during the next period.
    if next_high >= sellpoint:
        sell_price = sellpoint  # Sell at the sell point.
        kraken_sell_fee = amount_purchased * sell_price * 0.0016
        net_pnl = (amount_purchased * sell_price) - kraken_buy_fee - kraken_sell_fee
    else:
        sell_price = next_close  # If the sell point was not reached, sell at the close price of the next period.
        kraken_sell_fee = amount_purchased * sell_price * 0.0016
        net_pnl = (amount_purchased * sell_price) - kraken_buy_fee - kraken_sell_fee

    balance += net_pnl  # Update the balance with the net P&L from the trade.

    return net_pnl, balance  # Return the net profit or loss from the trade and the updated balance.

# Function to calculate P&L for each CSV file for a single set of trading models.
def calculate_pnl_per_set(input_dir, output_directory):
    df = pd.read_csv(input_dir)  # Read the CSV file containing the predictions for this set of models.
    total_pnl, end_balance = calculate_trades(starting_balance, df)  # Calculate trades, total P&L, and end balance.

    filename = os.path.splitext(os.path.basename(input_dir))[0]  # Get the filename without extension for output.
    end_results_df = pd.DataFrame({  # Create a DataFrame with the total P&L and end balance results.
        'Total_PnL': [total_pnl],
        'Ending_Balance': [end_balance],
    })
    output_file = os.path.join(output_directory, f"{filename}_end_results.csv")  # Define the path for the output CSV.
    end_results_df.to_csv(output_file, index=False)  # Save the results to a CSV file in the output directory.

# Function to calculate P&L for each set of models for all cryptocurrencies.
def calculate_pnl_all_sets(input_directory, output_directory):
    # If the output directory does not exist, create it.
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over all CSV files in the input directory.
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_directory, filename)  # Create full input file path.
            calculate_pnl_per_set(input_file, output_directory)  # Calculate P&L for this set of models.
            print(f"End results for {filename} saved.")  # Print a confirmation message for the user.

    print("-" * 30)  # Print a divider line for readability.

# Specify the input directory containing all CSV files with model predictions and the output directory for results.
input_directory = "./predictions_set"
output_directory = "./pnl_results"

# Call the function to calculate P&L for all sets of models for each cryptocurrency CSV file.
calculate_pnl_all_sets(input_directory, output_directory)

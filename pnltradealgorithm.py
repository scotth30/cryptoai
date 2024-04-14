import pandas as pd
import os

starting_balance = 100  # Starting balance in USD, used as initial amount for the trading simulation.
cryptocurrencies = ["ETH", "BCH", "WAVES", "SUI", "DOT", "XRP", "ADA", "KAVA", "SUSHI", "LINK", "UNI", "APE", "BTC"]
leverage_positions = ["25", "50", "100", "200"]

def calculate_trades(balance, df):
    total_pnl = 0
    total_pnl_leverage = {leverage: 0 for leverage in leverage_positions}  # Initialize total P&L for each leverage position
    balance_leverage = {}  # Initialize balance for each leverage position

    for i in range(8):
        close_col = f'close_{i}'
        high_col = f'high_{i}'
        low_col = f'low_{i}'

        if close_col in df.columns and high_col in df.columns and low_col in df.columns:
            predicted_close = df[close_col]
            predicted_high = df[high_col]
            predicted_low = df[low_col]
            
            for j in range(len(df) - 1):
                if predicted_close[j] > df['close'][j] * 1.01 and predicted_high[j] >= df['high'][j] * 1.01 and predicted_low[j] * 1.005 >= df['low'][j]:
                    predicted_range = predicted_high[j] - predicted_low[j]
                    buypoint = df['close'][j] + (predicted_range * 0.1)
                    sellpoint = df['close'][j] - (predicted_range * 0.1)

                    if j + 1 < len(df):
                        if buypoint >= df['low'][j+1]:
                            next_close = df['close'][j+1]
                            next_high = df['high'][j+1]
                            next_low = df['low'][j+1]
                            
                            pnl, balance = calculate_net_pnl(balance, buypoint, sellpoint, df['close'][j], next_high, next_close, next_low)
                            total_pnl += pnl

                            # Loop through each leverage position
                            for leverage in leverage_positions:
                                pnl_leverage, balance_leverage[leverage] = calculate_net_pnl_leveraged(balance, buypoint, sellpoint, df['close'][j], next_high, next_close, next_low, leverage)
                                total_pnl_leverage[leverage] += pnl_leverage

    return total_pnl, balance, total_pnl_leverage, balance_leverage

def calculate_net_pnl(balance, buypoint, sellpoint, current_close, next_high, next_close, next_low):
    amount_to_trade = starting_balance * 0.10  # Always trade 10% of the starting balance
    amount_purchased = amount_to_trade / buypoint
    kraken_buy_fee = amount_to_trade * 0.0016

    if next_high >= sellpoint:
        sell_price = sellpoint
        kraken_sell_fee = amount_purchased * sell_price * 0.0016
        total_purchase_cost = amount_purchased * buypoint
        net_pnl = (amount_purchased * sell_price) - kraken_buy_fee - kraken_sell_fee - total_purchase_cost
    else:
        sell_price = next_close
        kraken_sell_fee = amount_purchased * sell_price * 0.0026
        total_purchase_cost = amount_purchased * buypoint
        net_pnl = (amount_purchased * sell_price) - kraken_buy_fee - kraken_sell_fee - total_purchase_cost

    balance += net_pnl

    return net_pnl, balance 

def calculate_net_pnl_leveraged(balance, buypoint, sellpoint, current_close, next_high, next_close, next_low, leverage):
    amount_to_trade = starting_balance * 0.10  # Always trade 10% of the starting balance
    leverage_fee = amount_to_trade * 0.15
    amount_purchased = amount_to_trade / buypoint
    limit_loss = 0  # Default value for limit_loss
    if leverage == "200":
        limit_loss = buypoint * 0.005
        leverage_fee = amount_to_trade * 0.30
    elif leverage == "100":
        limit_loss = buypoint * 0.01
        leverage_fee = amount_to_trade * 0.15
    elif leverage == "50":
        limit_loss = buypoint * 0.02
        leverage_fee = amount_to_trade * 0.10
    elif leverage == "25":
        limit_loss = buypoint * 0.04
        leverage_fee = amount_to_trade * 0.05

    total_price = amount_to_trade + leverage_fee
    
    if next_low <= limit_loss:
        net_pnl = -total_price
        balance -= total_price
    else:
        if next_high >= sellpoint:
            sell_price = sellpoint
        else:
            sell_price = next_close

        total_purchase_cost = amount_to_trade * buypoint

        net_pnl = (amount_purchased * sell_price * int(leverage)) - leverage_fee - total_purchase_cost

        balance += net_pnl

    return net_pnl, balance

def calculate_pnl_per_set(input_dir, output_directory):
    df = pd.read_csv(input_dir)
    total_pnl, end_balance, total_pnl_leverage, balance_leverage = calculate_trades(starting_balance, df)

    filename = os.path.splitext(os.path.basename(input_dir))[0]
    end_results_df = pd.DataFrame({
        'Total_PnL': [total_pnl],  # Total P&L without leverage
        'Ending_Balance': [end_balance],
    })

    # Calculate and store P&L for each leverage position
    for leverage in leverage_positions:
        end_results_df[f'Total_PnL_{leverage}'] = [total_pnl_leverage[leverage]]

    output_file = os.path.join(output_directory, f"{filename}_end_results.csv")
    end_results_df.to_csv(output_file, index=False)


def calculate_pnl_all_sets(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_directory, filename)
            calculate_pnl_per_set(input_file, output_directory)
            print(f"End results for {filename} saved.")

    print("-" * 30)

# Specify the input directory containing all CSV files with model predictions and the output directory for results.
input_directory = "./predictions_set"
output_directory = "./pnl_results"

# Call the function to calculate P&L for all sets of models for each cryptocurrency CSV file.
calculate_pnl_all_sets(input_directory, output_directory)

from fbprophet import Prophet
import pandas as pd
import os
import glob
import dill as pickle  # Use dill in place of pickle for better compatibility with Prophet models

data_dir = '\\path\\to\\your\\data\\directory\\'
output_dir = '\\path\\to\\your\\output\\directory\\'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_file(filepath):
    print(f'Processing {filepath}')
    df = pd.read_parquet(filepath)
    df_prophet = df.rename(columns={'open_time': 'ds', 'close': 'y'})

    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    additional_features = ['open', 'high', 'low', 'volume', 'quote_asset_volume', 'number_of_trades',
                           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'SMA_7', 'SMA_30',
                           'daily_return', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'middle_BB',
                           'upper_BB', 'lower_BB', 'OBV', 'low_min', 'high_max', '%K', '%D', 'log_return',
                           'Price_Up', 'day_of_week', 'day_of_month']
    for feature in additional_features:
        model.add_regressor(feature)

    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=365)
    # Note: Ensure you add future values for regressors in the `future` dataframe

    forecast = model.predict(future)

    # Save the forecast
    forecast_output_path = os.path.join(output_dir, os.path.basename(filepath).replace('.parquet', '_forecast.csv'))
    forecast.to_csv(forecast_output_path, index=False)
    
    # Save the model components and plots for evaluation
    fig1 = model.plot(forecast)
    fig2 = model.plot_components(forecast)
    fig1.savefig(os.path.join(output_dir, os.path.basename(filepath).replace('.parquet', '_forecast_plot.png')))
    fig2.savefig(os.path.join(output_dir, os.path.basename(filepath).replace('.parquet', '_components_plot.png')))

    # Save the model using dill
    model_output_path = os.path.join(output_dir, os.path.basename(filepath).replace('.parquet', '_model.pkl'))
    with open(model_output_path, 'wb') as model_file:
        pickle.dump(model, model_file)

for filepath in glob.glob(os.path.join(data_dir, '*.parquet')):
    process_file(filepath)

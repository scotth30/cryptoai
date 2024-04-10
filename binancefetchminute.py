import requests
import csv
from datetime import datetime, timedelta
import concurrent.futures
import os
import logging
import zipfile

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
API_KEY = '57b1fda758f731788dbf90b8a3fb2a0157764b94079dbb349d0ca9ea630ecabf'

cryptocurrencies = [
    "BTC", "ETH", "LPT", "WAVES", "SUI", "DOT", "DOGE", "ADA", "ETC", "KAVA", "XRP", "BCH", "ARB", "EOS", "SUSHI", "LINK", "UNI", "APE"
]

def write_sorted_data_to_file(symbol, all_data):
    sorted_data = sorted(all_data, key=lambda x: x['timestamp'])
    filename = f'{symbol}_minute_data.csv'
    with open(filename, mode='w', newline='') as file:
        fieldnames = ['timestamp', 'high', 'low', 'open', 'close', 'volume', 'trades']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_data:
            writer.writerow({
                'timestamp': row['timestamp'],
                'high': row['high'],
                'low': row['low'],
                'open': row['open'],
                'close': row['close'],
                'volume': row['volume'],
                'trades': row['trades']
            })

def parse_csv_row(row):
    try:
        timestamp = int(row[0])
        high = float(row[1])
        low = float(row[2])
        open_price = float(row[3])
        close = float(row[4])
        volume = float(row[5])
        trades = int(row[8])  # Assuming trades is at index 6
        return {
            'timestamp': timestamp,
            'high': high,
            'low': low,
            'open': open_price,
            'close': close,
            'volume': volume,
            'trades': trades
        }
    except (ValueError, IndexError):
        logger.warning(f"Irregular row format: {row}")
        return None

def fetch_minute_data_for_period(symbol, start_ts, end_ts):
    base_url = 'https://data.binance.vision/data/spot/daily/klines/'
    date_format = '%Y-%m-%d'
    all_data = []
    
    current_date = start_ts
    while current_date < end_ts:
        date_str = current_date.strftime(date_format)
        url = f'{base_url}{symbol}USDT/1m/{symbol}USDT-1m-{date_str}.zip'
        
        response = requests.get(url)
        if response.status_code == 200:
            # Assuming you have downloaded the zip file to a directory named 'data'
            zip_path = f'data/{symbol}_data.zip'
            extract_path = f'data/{symbol}/'
            
            # Check if the directory exists, if not, create it
            os.makedirs(extract_path, exist_ok=True)
            
            with open(zip_path, 'wb') as zip_file:
                zip_file.write(response.content)
            
            # Unzipping the downloaded file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            # Reading the CSV file from the extracted zip
            csv_file = f'{extract_path}{symbol}USDT-1m-{date_str}.csv'
            with open(csv_file, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    parsed_row = parse_csv_row(row)
                    if parsed_row is not None:
                        all_data.append(parsed_row)
        else:
            logger.error(f"Error downloading data for {symbol} on {date_str}. Status code: {response.status_code}")
        
        current_date += timedelta(days=1)
    
    write_sorted_data_to_file(symbol, all_data)

def fetch_data_for_symbol(symbol):
    logger.info(f"Starting fetch for {symbol} from Binance")
    start_date = datetime.now() - timedelta(days=120)
    end_date = datetime.now() - timedelta(days=1)  # Yesterday
    fetch_minute_data_for_period(symbol, start_date, end_date)

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(fetch_data_for_symbol, cryptocurrencies)

import pandas as pd
import numpy as np
import os
from datetime import timedelta

# Utility function to generate walk-forward validation periods
def generate_periods(start_date, end_date):
    """Generate periods for walk-forward validation with 1-year training and 1-month test."""
    periods = []
    current_start = pd.to_datetime(start_date)
    final_start = pd.to_datetime(end_date)
    
    while current_start <= final_start:
        train_start = current_start
        train_end = train_start + pd.DateOffset(years=1) - timedelta(days=1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=1) - timedelta(days=1)
        
        if test_end > final_start:
            break

        periods.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })
        
        current_start += pd.DateOffset(months=1)
    
    return periods

# Utility function to slice data by periods
def slice_data_by_period(data, periods, date_col):
    """Slice data by the given periods."""
    slices = []
    for period in periods:
        train_data = data[(data[date_col] >= period['train_start']) & (data[date_col] <= period['train_end'])]
        test_data = data[(data[date_col] >= period['test_start']) & (data[date_col] <= period['test_end'])]
        slices.append((train_data, test_data))
    return slices

def process_file(file_path, output_path):
    """Process the CSV file to calculate price changes and up/down movements."""
    df = pd.read_csv(file_path)

    # Convert 'Gmt time' column to datetime
    df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')

    # Set 'Gmt time' as index
    df.set_index('Gmt time', inplace=True)

    # Calculate price change and determine up/down movements
    df['Price Change'] = df['Close'].diff()
    df['Up/Down'] = df['Price Change'].shift(-1).apply(lambda x: 'Up' if x > 0 else ('Down' if x < 0 else 'No Change'))
    df.drop(columns=['Price Change'], inplace=True)

    # Save the processed DataFrame to a CSV file
    df.to_csv(output_path)

# Get the base directory from the environment variable
base_dir = os.environ.get('DYNAMIC_PORTFOLIO_ALLOCATION', os.getcwd())

# Define file paths for EURUSD and GBPUSD
eurusd_file_path = os.path.join(base_dir, 'Data_Prep', 'EURUSD_Candlestick_1_Hour_ASK_30.04.2020-31.12.2023.csv')
eurusd_output_path = os.path.join(base_dir, 'Feature_Engineering', 'EURUSD_hourly_with_changes.csv')

gbpusd_file_path = os.path.join(base_dir, 'Data_Prep', 'GBPUSD_Candlestick_1_Hour_ASK_30.04.2020-31.12.2023.csv')
gbpusd_output_path = os.path.join(base_dir, 'Feature_Engineering', 'GBPUSD_hourly_with_changes.csv')

# Process the files and save the results
process_file(eurusd_file_path, eurusd_output_path)
process_file(gbpusd_file_path, gbpusd_output_path)

import os
import pandas as pd

def combine_csv_files(directory, file_suffix):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(file_suffix)]
    return pd.concat((pd.read_csv(file) for file in files), ignore_index=True) if files else pd.DataFrame()

def resample_to_daily(data, time_column='time', value_column='net_pnl_change'):
    """Resample the PnL data to daily frequency and aggregate by strategy_name."""
    data[time_column] = pd.to_datetime(data[time_column])
    daily_data = (data.set_index(time_column)
                    .groupby('strategy_name')
                    .resample('D')[value_column]
                    .sum()
                    .groupby(level=0)
                    .cumsum()
                    .reset_index(name='net_pnl'))
    return daily_data

def main():
    base_out = os.path.dirname(os.path.abspath(__file__))
    base_in = os.environ.get('DYNAMIC_PORTFOLIO_ALLOCATION', os.getcwd())
    data_directory = os.path.join(base_in, 'Strategy_Data', 'forex-dynamic-portfolio-allocation')
    output_directory = os.path.join(base_out)
    os.makedirs(output_directory, exist_ok=True)

    combined_position_data = combine_csv_files(data_directory, 'positions.csv')
    if not combined_position_data.empty:
        combined_position_data.to_csv(os.path.join(output_directory, 'combined_position_data.csv'), index=False)
        print("Combined position data saved.")

    combined_pnl_data = combine_csv_files(data_directory, 'pnl.csv')
    if not combined_pnl_data.empty:
        combined_pnl_data.to_csv(os.path.join(output_directory, 'combined_pnl_data.csv'), index=False)
        print("Combined pnl data saved.")

if __name__ == "__main__":
    main()
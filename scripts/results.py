import os
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from tqdm import tqdm

class AllocationExtrapolator:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.df_hourly_pnl = None
        self.results_df = None
        # Define the cutoff dates
        self.start_cutoff = pd.to_datetime('2022-07-01 00:00:00+00:00', utc=True)
        self.end_cutoff = pd.to_datetime('2023-12-31 23:00:00+00:00', utc=True)

    def load_combined_hourly_data(self):
        """Loads the combined hourly PnL data and renames 'time' to 'Date'."""
        file_path = os.path.join(self.base_dir, 'combined_hourly_pnl_data_updated.csv')
        self.df_hourly_pnl = pd.read_csv(file_path)
        self.df_hourly_pnl['Date'] = pd.to_datetime(self.df_hourly_pnl['time'], utc=True)
        self.df_hourly_pnl.drop(columns=['time'], inplace=True)

        self.df_hourly_pnl = self.df_hourly_pnl[(self.df_hourly_pnl['Date'] >= self.start_cutoff) & 
                                                (self.df_hourly_pnl['Date'] <= self.end_cutoff)]

    def load_results_data(self):
        """Loads the results.csv file and ensures the 'Date' column is datetime."""
        file_path = os.path.join(self.base_dir, 'results.csv')
        self.results_df = pd.read_csv(file_path)
        self.results_df['Date'] = pd.to_datetime(self.results_df['Date'], utc=True)

        self.results_df = self.results_df[(self.results_df['Date'] >= self.start_cutoff) & 
                                          (self.results_df['Date'] <= self.end_cutoff)]

    def extrapolate_to_hourly(self):
        """Extrapolates each row of the results.csv to all the hours in the corresponding month."""
        hourly_data = []

        for _, row in tqdm(self.results_df.iterrows(), total=len(self.results_df), desc="Extrapolating to Hourly Data"):
            start_date = row['Date']
            end_date = (start_date + MonthEnd(0)).replace(hour=23, minute=0, second=0)
            hourly_range = pd.date_range(start=start_date, end=end_date, freq='h', tz='UTC')

            strategy_data = pd.DataFrame({
                'Date': hourly_range,
                'strategy_name': row['strategy_name'],
                'y_actual': row['y_actual_MLP'],
                'y_pred_MLP': row['y_pred_MLP'],
                'y_pred_SGD': row['y_pred_SGD'],
                'y_pred_ABR': row['y_pred_ABR'],
            })
            hourly_data.append(strategy_data)

        hourly_data = pd.concat(hourly_data, ignore_index=True)
        hourly_data = hourly_data[(hourly_data['Date'] >= self.start_cutoff) & 
                                  (hourly_data['Date'] <= self.end_cutoff)]

        return hourly_data

    def merge_with_hourly_pnl(self, hourly_data):
        """Merges the extrapolated hourly data with the combined_hourly_pnl_data_updated.csv."""
        merged_df = pd.merge(self.df_hourly_pnl, hourly_data, on=['Date', 'strategy_name'], how='left')
        merged_df.fillna(0, inplace=True)
        output_path = os.path.join(self.base_dir, 'Allocation.csv')
        merged_df.to_csv(output_path, index=False)

    def run(self):
        """Executes the entire process."""
        self.load_combined_hourly_data()
        self.load_results_data()
        hourly_data = self.extrapolate_to_hourly()
        self.merge_with_hourly_pnl(hourly_data)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    analyzer = AllocationExtrapolator(base_dir)
    analyzer.run()

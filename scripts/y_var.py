import os
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
from tqdm import tqdm

class StrategyAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.df = None
        self.top_10_results = None
        self.data = None

    def load_data(self):
        file_path = os.path.join(self.base_dir, 'combined_hourly_pnl_data_updated.csv')
        self.df = pd.read_csv(file_path)
        self.df['Date'] = pd.to_datetime(self.df['time'])

    def load_full_csv(self):
        file_path = os.path.join(self.base_dir, 'full.csv')
        full_df = pd.read_csv(file_path)
        full_df['Date'] = pd.to_datetime(full_df['Date'])
        return full_df

    def calculate_net_pnl_change(self):
        return self.df.groupby(['strategy_name', self.df['Date'].dt.to_period('M')])['net_pnl_change'].sum().reset_index().rename(columns={'Date': 'Period'})

    def calculate_top_10(self):
        result_df = self.calculate_net_pnl_change()
        top_10_results = pd.DataFrame()
        
        # Adding progress bar for top 10 calculation
        for period in tqdm(result_df['Period'].unique(), desc="Calculating Top 10 Strategies per Period"):
            period_data = result_df[result_df['Period'] == period]
            top_10_period = period_data.nlargest(10, 'net_pnl_change')
            
            # Assign rank based on net_pnl_change
            top_10_period['Rank'] = top_10_period['net_pnl_change'].rank(ascending=True, method='first').astype(int)
            
            # Shift the Period back by 1 month
            top_10_period['Shifted_Period'] = (top_10_period['Period'] - 1).apply(lambda x: x.strftime('%Y-%m'))

            top_10_results = pd.concat([top_10_results, top_10_period])

        self.top_10_results = top_10_results

    def extrapolate_to_hourly(self):
        hourly_data = pd.DataFrame()

        # Adding progress bar for extrapolation
        for _, row in tqdm(self.top_10_results.iterrows(), total=len(self.top_10_results), desc="Extrapolating to Hourly Data"):
            period = str(row['Shifted_Period'])
            start_date = pd.to_datetime(period + '-01 00:00:00+00:00')
            end_date = (start_date + MonthEnd(1)).replace(hour=23, minute=0, second=0)
            
            hourly_range = pd.date_range(start=start_date, end=end_date, freq='h', tz='UTC')
            strategy_data = pd.DataFrame({
                'Date': hourly_range,
                'strategy_name': row['strategy_name'],
                'Shifted_Period': row['Shifted_Period'],
                'net_pnl_change': row['net_pnl_change'],
                'Rank': row['Rank']  # Add rank to hourly data
            })

            hourly_data = pd.concat([hourly_data, strategy_data])

        output_path = os.path.join(self.base_dir, 'Top_N_Results', 'top_10_hourly_M.csv')
        hourly_data.to_csv(output_path, index=False)

        return hourly_data

    def merge_with_full(self, hourly_data):
        full_df = self.load_full_csv()

        # Merge the hourly top 10 with full.csv on Date and strategy_name
        merged_df = pd.merge(full_df, hourly_data, on=['Date', 'strategy_name'], how='left')
        merged_df['Rank'] = merged_df['Rank'].fillna(0)
        # Save the merged result
        output_path = os.path.join(self.base_dir, 'Top_N_Results', 'full.csv')
        merged_df.to_csv(output_path, index=False)

    def save_top_10_results(self):
        output_dir = os.path.join(self.base_dir, 'Top_N_Results')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "top_10_M.csv")
        self.top_10_results.to_csv(output_path, index=False)

    def run_analysis(self):
        self.load_data()
        self.calculate_top_10()
        self.save_top_10_results()
        hourly_data = self.extrapolate_to_hourly()
        self.merge_with_full(hourly_data)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    analyzer = StrategyAnalyzer(base_dir)
    analyzer.run_analysis()

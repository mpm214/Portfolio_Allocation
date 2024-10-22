import os
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm

class StrategyMetricsCalculator:
    def __init__(self, pnl_path, start_date='2021-06-01', end_date='2024-01-01'):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.pnl_path = pnl_path
        self.pnl_data = None
        self.start_date = pd.to_datetime(start_date, utc=True)
        self.end_date = pd.to_datetime(end_date, utc=True)

    def load_data(self):
        """Load PnL data from CSV, resample to daily, and parse dates"""
        self.pnl_data = pd.read_csv(self.pnl_path)
        self.pnl_data['time'] = pd.to_datetime(self.pnl_data['time'], utc=True)
        
        # Resample to daily for net_pnl using mean, and net_pnl_change using sum
        self.pnl_data = self.pnl_data.rename(columns={'time': 'Date'})

        # Fill missing start and end dates for each strategy
        self.backfill_and_forward_fill()

    def backfill_and_forward_fill(self):
        """Backfill and forward fill the missing dates for each strategy"""
        strategies = self.pnl_data['strategy_name'].unique()
        filled_data = []

        for strategy in strategies:
            strategy_data = self.pnl_data[self.pnl_data['strategy_name'] == strategy].copy()
            strategy_data = strategy_data.set_index('Date')

            # Create a date range from start_date to end_date
            full_date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='h', tz='UTC')

            # Reindex to ensure all dates are present
            strategy_data = strategy_data.reindex(full_date_range)

            # Forward fill the last available data to the end_date
            strategy_data.ffill(inplace=True)

            # Reset index and add strategy_name back
            strategy_data = strategy_data.reset_index().rename(columns={'index': 'Date'})
            strategy_data['strategy_name'] = strategy

            filled_data.append(strategy_data)

        # Concatenate all strategies back into one DataFrame
        self.pnl_data = pd.concat(filled_data, ignore_index=True)

    def calculate_training_metrics(self):
        """Calculate metrics for training data as a whole for each strategy"""
        train_start = self.start_date
        train_end = self.end_date

        training_results = []

        for strategy in tqdm(self.pnl_data['strategy_name'].unique(), desc="Calculating Training Metrics"):
            strategy_data = self.pnl_data[(self.pnl_data['strategy_name'] == strategy) & 
                                          (self.pnl_data['Date'] >= train_start) & 
                                          (self.pnl_data['Date'] <= train_end)].copy()

            strategy_data = self.calculate_metrics(strategy_data)
            strategy_data = strategy_data.bfill()

            training_results.append(strategy_data)

        return pd.concat(training_results, ignore_index=True)

    def calculate_metrics(self, period_df):
        """Calculate financial metrics for each strategy"""
        # Rolling 30-day Standard Deviation
        period_df['rolling_std_30'] = period_df.groupby('strategy_name')['net_pnl_change'].rolling(window=720).std().reset_index(level=0, drop=True)

        # First and Second Derivative of rolling_std_30
        period_df['std_ROC'] = period_df.groupby('strategy_name')['rolling_std_30'].transform(lambda x: x.diff())
        period_df['std_Momentum'] = period_df.groupby('strategy_name')['std_ROC'].transform(lambda x: x.diff())

        # Rolling 30-day Average Return
        period_df['rolling_avg_30'] = period_df.groupby('strategy_name')['net_pnl_change'].rolling(window=720).mean().reset_index(level=0, drop=True)

        # First and Second Derivative of rolling_avg_30
        period_df['PnL_Avg_ROC'] = period_df.groupby('strategy_name')['rolling_avg_30'].transform(lambda x: x.diff())
        period_df['PnL_Avg_Momentum'] = period_df.groupby('strategy_name')['PnL_Avg_ROC'].transform(lambda x: x.diff())

        # Sharpe Ratio without Risk-Free Rate
        period_df['sharpe_ratio'] = period_df['rolling_avg_30'] / period_df['rolling_std_30']
        period_df['sharpe_ratio'] = period_df['sharpe_ratio'].replace([np.inf, -np.inf], 0)

        # First and Second Derivative of sharpe_ratio
        period_df['Sharpe_ROC'] = period_df.groupby('strategy_name')['sharpe_ratio'].transform(lambda x: x.diff())
        period_df['Sharpe_Momentum'] = period_df.groupby('strategy_name')['Sharpe_ROC'].transform(lambda x: x.diff())

        # Rolling Peak and Drawdown
        period_df['rolling_Peak_30'] = period_df.groupby('strategy_name')['net_pnl'].rolling(window=720).max().reset_index(level=0, drop=True)
        period_df['Drawdown'] = (period_df['net_pnl'] - period_df['rolling_Peak_30']) / period_df['rolling_Peak_30']
        period_df['Drawdown'] = period_df['Drawdown'].replace([np.inf, -np.inf], 0)

        # Long/Short Indicator
        period_df['Long_Short'] = period_df['strategy_name'].apply(lambda x: 1 if x.startswith('L_') else -1)

        # EURUSD/GBPUSD Indicator
        period_df['CCY'] = period_df['strategy_name'].apply(lambda x: "EURUSD" if '_EURUSD' in x else "GBPUSD")

        # Recovery Time Features
        period_df['rolling_Peak_Recovery_Days'] = (period_df['rolling_Peak_30'] == period_df['net_pnl']).astype(int)
        period_df['Recovery_Time'] = period_df.groupby('strategy_name')['rolling_Peak_Recovery_Days'].transform(self.calculate_recovery_time)
        period_df['Rolling_Max_Recovery_Time'] = period_df.groupby('strategy_name')['Recovery_Time'].rolling(window=720).max().reset_index(level=0, drop=True)

        # 30-Day Slope and Momentum
        period_df['PnL_slope_30'] = period_df.groupby('strategy_name')['net_pnl'].transform(lambda x: (x - x.shift(696)) / 720)
        period_df['PnL_momentum_30'] = period_df.groupby('strategy_name')['PnL_slope_30'].transform(lambda x: x.diff())

        # Time-Based Features
        period_df['month'] = period_df['Date'].dt.month
        period_df['day_of_week'] = period_df['Date'].dt.dayofweek
        period_df['day_of_month'] = period_df['Date'].dt.day

        return period_df

    @staticmethod
    def calculate_recovery_time(series):
        """Calculate the number of days to recover from drawdown"""
        recovery_times = []
        count = 0
        for value in series:
            if value == 1:
                recovery_times.append(count)
                count = 0
            else:
                recovery_times.append(count)
                count += 1
        return recovery_times

    def run(self):
        """Run the data processing and metrics calculation"""
        self.load_data()
        
        # Calculate metrics for training data
        training_metrics = self.calculate_training_metrics()

        # Save to CSV
        output_path = os.path.join(self.base_dir, 'strategy_hourly_performance.csv')
        training_metrics.to_csv(output_path, index=False)
        print(f"Strategy metrics saved to: {output_path}")

if __name__ == "__main__":
    pnl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined_hourly_pnl_data_updated.csv')
    strategy_metrics_calculator = StrategyMetricsCalculator(pnl_path)
    strategy_metrics_calculator.run()

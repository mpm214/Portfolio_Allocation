import os
import pandas as pd
import numpy as np
from tqdm import tqdm

class StrategyRatiosCalculator:
    def __init__(self, trades_path, eurusd_path, gbpusd_path):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.trades_path = trades_path
        self.eurusd_path = eurusd_path
        self.gbpusd_path = gbpusd_path
        self.start_date = '2021-06-01'
        self.initial_end_date = '2022-06-01'
        self.final_end_date = '2024-01-01'

    def load_data(self):
        """Load EURUSD and GBPUSD datasets."""
        self.eurusd_df = pd.read_csv(self.eurusd_path)
        self.eurusd_df['Time'] = pd.to_datetime(self.eurusd_df['Gmt time'], utc=True)
        self.gbpusd_df = pd.read_csv(self.gbpusd_path)
        self.gbpusd_df['Time'] = pd.to_datetime(self.gbpusd_df['Gmt time'], utc=True)

    def calculate_ratios(self, data, up_value, down_value, is_long):
        """Calculate strategy performance ratios."""
        if is_long:
            open_up = data[(data['Trade Status'] == 'Open') & (data['Up/Down'] == up_value)].shape[0]
            open_down = data[(data['Trade Status'] == 'Open') & (data['Up/Down'] == down_value)].shape[0]
            total_up = data[data['Up/Down'] == up_value].shape[0]
            total_down = data[data['Up/Down'] == down_value].shape[0]
            precision = open_up / total_up if total_up > 0 else None
            downside_ratio = open_down / total_down if total_down > 0 else None
            combined_ratio = open_up / (open_up + open_down) if (open_up + open_down) > 0 else None
            sum_close_up = data[(data['Trade Status'] == 'Open') & (data['Up/Down'] == up_value)]['Close_Change'].sum()
            sum_close_down = abs(data[(data['Trade Status'] == 'Open') & (data['Up/Down'] == down_value)]['Close_Change'].sum())
            total_sum_close_up = data[data['Up/Down'] == up_value]['Close_Change'].sum()
            total_sum_close_down = abs(data[data['Up/Down'] == down_value]['Close_Change'].sum())
            precision_magnitude = sum_close_up / total_sum_close_up if total_sum_close_up > 0 else None
            downside_ratio_magnitude = sum_close_down / total_sum_close_down if total_sum_close_down > 0 else None
            combined_ratio_magnitude = sum_close_up / (sum_close_up + sum_close_down) if (sum_close_up + sum_close_down) > 0 else None
        else:
            open_down = data[(data['Trade Status'] == 'Open') & (data['Up/Down'] == down_value)].shape[0]
            open_up = data[(data['Trade Status'] == 'Open') & (data['Up/Down'] == up_value)].shape[0]
            total_down = data[data['Up/Down'] == down_value].shape[0]
            total_up = data[data['Up/Down'] == up_value].shape[0]
            precision = open_down / total_down if total_down > 0 else None
            downside_ratio = open_up / total_up if total_up > 0 else None
            combined_ratio = open_down / (open_down + open_up) if (open_down + open_up) > 0 else None
            sum_close_down = abs(data[(data['Trade Status'] == 'Open') & (data['Up/Down'] == down_value)]['Close_Change'].sum())
            sum_close_up = data[(data['Trade Status'] == 'Open') & (data['Up/Down'] == up_value)]['Close_Change'].sum()
            total_sum_close_down = abs(data[data['Up/Down'] == down_value]['Close_Change'].sum())
            total_sum_close_up = data[data['Up/Down'] == up_value]['Close_Change'].sum()
            precision_magnitude = sum_close_down / total_sum_close_down if total_sum_close_down > 0 else None
            downside_ratio_magnitude = sum_close_up / total_sum_close_up if total_sum_close_up > 0 else None
            combined_ratio_magnitude = sum_close_down / (sum_close_down + sum_close_up) if (sum_close_down + sum_close_up) > 0 else None

        return precision, downside_ratio, combined_ratio, precision_magnitude, downside_ratio_magnitude, combined_ratio_magnitude

    def calculate_strategy_ratios(self):
        """Calculate strategy ratios for each strategy, applying the initial year and monthly expansions."""
        results = []

        # Load the trades data once to optimize the process
        trades_df = pd.read_csv(self.trades_path)
        trades_df['Time'] = pd.to_datetime(trades_df['Time'], utc=True)

        # Define the start, initial end, and final end dates
        start_date = pd.to_datetime(self.start_date, utc=True)
        initial_end_date = pd.to_datetime(self.initial_end_date, utc=True)
        final_end_date = pd.to_datetime(self.final_end_date, utc=True)

        # Generate hourly date range from start to final end date
        hourly_date_range = pd.date_range(start=start_date, end=final_end_date, freq='h', tz='UTC')

        # Get unique strategies
        strategies = trades_df['Strategy'].unique()

        # Create a DataFrame to hold results for every hour and strategy
        strategy_hourly_df = pd.MultiIndex.from_product([hourly_date_range, strategies], names=['Date', 'Strategy']).to_frame(index=False)

        # Calculate for the first year (2021-06-01 to 2022-06-01)
        initial_window_df = trades_df[(trades_df['Time'] >= start_date) & (trades_df['Time'] <= initial_end_date)]

        # Initialize progress bar for the first year calculation
        with tqdm(total=len(strategies), desc="Calculating First Year Ratios", unit="strategy") as pbar:
            for strategy in strategies:
                asset_data = self.eurusd_df if "EURUSD" in strategy else self.gbpusd_df
                is_long = strategy.startswith('L')

                # Get data for the strategy within the first year
                strategy_data = initial_window_df[initial_window_df['Strategy'] == strategy].merge(asset_data, on='Time')
                strategy_ratios = self.calculate_ratios(strategy_data, 'Up', 'Down', is_long)

                # Apply the calculated ratios to all hourly rows within the first year
                strategy_hourly_df.loc[
                    (strategy_hourly_df['Strategy'] == strategy) & (strategy_hourly_df['Date'] <= initial_end_date),
                    ['Precision', 'Downside_Ratio', 'Combined_Ratio', 'Precision_Magnitude', 'Downside_Ratio_Magnitude', 'Combined_Ratio_Magnitude']
                ] = strategy_ratios

                # Update the progress bar
                pbar.update(1)

        # After first year, calculate for each month-end and expand the window
        month_end_dates = pd.date_range(start=initial_end_date, end=final_end_date, freq='ME', tz='UTC')

        # Initialize last month-end to initial_end_date
        last_month_end = initial_end_date

        # Initialize progress bar for monthly expanding calculations
        with tqdm(total=len(month_end_dates), desc="Calculating Monthly Expanding Ratios", unit="month") as pbar:
            for month_end in month_end_dates:
                # Define the expanded window (start date to current month-end)
                expanded_window_df = trades_df[(trades_df['Time'] >= start_date) & (trades_df['Time'] < (month_end + pd.DateOffset(days=1)))]

                # Iterate through each strategy
                for strategy in strategies:
                    asset_data = self.eurusd_df if "EURUSD" in strategy else self.gbpusd_df
                    is_long = strategy.startswith('L')

                    # Get data for the strategy within the expanding window
                    strategy_data = expanded_window_df[expanded_window_df['Strategy'] == strategy].merge(asset_data, on='Time')
                    strategy_ratios = self.calculate_ratios(strategy_data, 'Up', 'Down', is_long)

                    # Assign the calculated values to all hourly rows between last month-end and current month-end
                    strategy_hourly_df.loc[
                        (strategy_hourly_df['Strategy'] == strategy) &
                        (strategy_hourly_df['Date'] > last_month_end) &
                        (strategy_hourly_df['Date'] < (month_end + pd.DateOffset(days=1))),
                        ['Precision', 'Downside_Ratio', 'Combined_Ratio', 'Precision_Magnitude', 'Downside_Ratio_Magnitude', 'Combined_Ratio_Magnitude']
                    ] = strategy_ratios

                # Update last_month_end to the current month-end for the next iteration
                last_month_end = month_end

                # Update the progress bar
                pbar.update(1)

        # Save results to CSV
        output_path = os.path.join(self.base_dir, 'strategy_effectiveness.csv')
        strategy_hourly_df.to_csv(output_path, index=False)
        print(f"Strategy ratios saved to: {output_path}")

    def run(self):
        """Run the complete ratio calculation process."""
        self.load_data()
        self.calculate_strategy_ratios()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    trades_path = os.path.join(base_dir, 'open_trades_times_by_strategy.csv')
    eurusd_path = os.path.join(base_dir, 'EURUSD_hourly_with_changes.csv')
    gbpusd_path = os.path.join(base_dir, 'GBPUSD_hourly_with_changes.csv')

    strategy_ratios_calculator = StrategyRatiosCalculator(trades_path, eurusd_path, gbpusd_path)
    strategy_ratios_calculator.run()

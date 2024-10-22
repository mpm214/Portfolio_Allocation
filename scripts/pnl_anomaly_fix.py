import os
import pandas as pd

class HourlyPnLCalculator:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.file_path = os.path.join(self.base_dir, 'combined_position_data_updated.csv')
        self.data = None
        self.results = []

    def load_data(self):
        self.data = pd.read_csv(self.file_path, parse_dates=['open_time', 'close_time'])
        if self.data['open_time'].dt.tz is None:
            self.data['open_time'] = self.data['open_time'].dt.tz_localize('UTC')
        if self.data['close_time'].dt.tz is None:
            self.data['close_time'] = self.data['close_time'].dt.tz_localize('UTC')
        self.data.sort_values(by=['strategy_name', 'open_time'], inplace=True)
        self.data['net_pnl_change'] = self.data['net_pnl']
        self.data['net_pnl'] = self.data.groupby('strategy_name')['net_pnl_change'].cumsum()

    def calculate_hourly_returns(self):
        for strategy_name, group in self.data.groupby('strategy_name'):
            start_time = group['open_time'].min().floor('h')
            end_time = group['close_time'].max().ceil('h')
            hourly_times = pd.date_range(start=start_time, end=end_time, freq='h', tz='UTC')

            hourly_returns = pd.DataFrame(hourly_times, columns=['time'])
            hourly_returns['strategy_name'] = strategy_name
            hourly_returns['net_pnl'] = 0.0
            hourly_returns['net_pnl_change'] = 0.0

            last_cumulative = 0.0

            for idx, time in enumerate(hourly_returns['time']):
                valid_trades = group[group['close_time'] <= time]
                if not valid_trades.empty:
                    last_valid_trade = valid_trades.iloc[-1]
                    current_cumulative = last_valid_trade['net_pnl']
                    hourly_returns.at[idx, 'net_pnl'] = current_cumulative
                    if idx == 0 or current_cumulative != last_cumulative:
                        hourly_returns.at[idx, 'net_pnl_change'] = last_valid_trade['net_pnl_change']
                    last_cumulative = current_cumulative
                else:
                    if idx > 0:
                        hourly_returns.at[idx, 'net_pnl'] = last_cumulative

            self.results.append(hourly_returns)

    def save_results(self):
        final_results = pd.concat(self.results, ignore_index=True)
        output_path = self.file_path.replace('combined_position_data_updated.csv', 'combined_hourly_pnl_data_updated.csv')
        final_results.to_csv(output_path, index=False)
        print(f"PnL data saved to {output_path}")

    def run(self):
        self.load_data()
        self.calculate_hourly_returns()
        self.save_results()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    calculator = HourlyPnLCalculator(base_dir)
    calculator.run()
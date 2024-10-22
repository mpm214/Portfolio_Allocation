import os
import pandas as pd

class OpenTradeTimesByStrategy:
    def __init__(self, trades_path, start_date='2021-06-01', end_date='2024-01-01'):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.trades_path = trades_path
        self.start_date = start_date
        self.end_date = end_date
        self.trades_df = None

    def load_data(self):
        self.trades_df = pd.read_csv(self.trades_path)
        self.trades_df['open_time'] = pd.to_datetime(self.trades_df['open_time'], errors='coerce')
        self.trades_df['close_time'] = pd.to_datetime(self.trades_df['close_time'], errors='coerce')
        # Convert to UTC if timezone is not set
        if self.trades_df['open_time'].dt.tz is None:
            self.trades_df['open_time'] = self.trades_df['open_time'].dt.tz_localize('UTC')
        else:
            self.trades_df['open_time'] = self.trades_df['open_time'].dt.tz_convert('UTC')
        if self.trades_df['close_time'].dt.tz is None:
            self.trades_df['close_time'] = self.trades_df['close_time'].dt.tz_localize('UTC')
        else:
            self.trades_df['close_time'] = self.trades_df['close_time'].dt.tz_convert('UTC')

    def open_trade_times_by_strategy(self):
        period_start, period_end = pd.to_datetime(self.start_date).tz_localize('UTC'), pd.to_datetime(self.end_date).tz_localize('UTC')
        period_df = self.trades_df[(self.trades_df['open_time'] >= period_start) & (self.trades_df['close_time'] <= period_end)]
        hourly_index = pd.date_range(period_start, period_end, freq='h', tz='UTC')
        result_dfs = []

        for strategy in period_df['strategy_name'].unique():
            strategy_df = period_df[period_df['strategy_name'] == strategy]
            strategy_result_df = pd.DataFrame(index=hourly_index)
            strategy_result_df.index.name = 'Time'
            strategy_result_df[['Trade Status', 'Trade Open/Close']] = 'No Trade'

            for _, trade in strategy_df.iterrows():
                open_hour = trade['open_time'].floor('h')
                close_hour = trade['close_time'].floor('h')
                strategy_result_df.loc[(strategy_result_df.index >= open_hour) & (strategy_result_df.index <= close_hour), 'Trade Status'] = 'Open'
                strategy_result_df.loc[open_hour, 'Trade Open/Close'] = 'Open Trade'
                strategy_result_df.loc[close_hour, 'Trade Open/Close'] = 'Close Trade'

            strategy_result_df['Strategy'] = strategy
            result_dfs.append(strategy_result_df)

        trades_processed_df = pd.concat(result_dfs).reset_index()
        output_path = os.path.join(self.base_dir, f'open_trades_times_by_strategy.csv')
        trades_processed_df.to_csv(output_path, index=False)
        print(f"Open trades times by strategy from {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')} saved to: {output_path}")

    def run(self):
        self.load_data()
        self.open_trade_times_by_strategy()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    trades_path = os.path.join(base_dir, 'combined_position_data_updated.csv')
    open_trade_processor = OpenTradeTimesByStrategy(trades_path)
    open_trade_processor.run()
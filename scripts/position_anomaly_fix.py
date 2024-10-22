import os
import pandas as pd

class PositionDataFixer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data = None
        self.filtered_data = None
        self.candlestick_ask = None
        self.candlestick_bid = None
        self.tolerance = 0.01

    def load_data(self):
        file_path = os.path.join(self.base_dir, 'combined_position_data.csv')
        self.data = pd.read_csv(file_path)
        self.data['open_time'] = pd.to_datetime(self.data['open_time']).dt.tz_localize('UTC')
        self.data['close_time'] = pd.to_datetime(self.data['close_time']).dt.tz_localize('UTC')

    def filter_data(self):
        start_date = pd.Timestamp('2022-10-01').tz_localize('UTC')
        end_date = pd.Timestamp('2022-10-10').tz_localize('UTC')
        pair_filter = 'GBP/USD'
        net_pnl_threshold = 700

        self.filtered_data = self.data[
            (((self.data['open_time'] >= start_date) & (self.data['open_time'] <= end_date)) |
             ((self.data['close_time'] >= start_date) & (self.data['close_time'] <= end_date))) &
            (self.data['pair'] == pair_filter) &
            ((self.data['net_pnl'] > net_pnl_threshold) | (self.data['net_pnl'] < -net_pnl_threshold))
        ]

    def load_candlestick_data(self):
        ask_path = os.path.join(self.base_dir, 'GBPUSD_Candlestick_1_M_ASK_30.09.2022-09.10.2022.csv')
        bid_path = os.path.join(self.base_dir, 'GBPUSD_Candlestick_1_M_BID_30.09.2022-09.10.2022.csv')
        self.candlestick_ask = pd.read_csv(ask_path)
        self.candlestick_bid = pd.read_csv(bid_path)

        self.candlestick_ask['Gmt time'] = pd.to_datetime(self.candlestick_ask['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f').dt.tz_localize('UTC')
        self.candlestick_bid['Gmt time'] = pd.to_datetime(self.candlestick_bid['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f').dt.tz_localize('UTC')

    def merge_price_data(self):
        self.filtered_data = self.filtered_data.merge(
            self.candlestick_ask[['Gmt time', 'Close']],
            left_on='open_time',
            right_on='Gmt time',
            how='left'
        ).rename(columns={'Close': 'close_price_at_open_time'}).drop(columns=['Gmt time'])

        self.filtered_data = self.filtered_data.merge(
            self.candlestick_bid[['Gmt time', 'Close']],
            left_on='close_time',
            right_on='Gmt time',
            how='left'
        ).rename(columns={'Close': 'close_price_at_close_time'}).drop(columns=['Gmt time'])

    def adjust_prices(self):
        self.filtered_data.loc[
            (self.filtered_data['close_price_at_open_time'] > self.filtered_data['open_price'] + self.tolerance) |
            (self.filtered_data['close_price_at_open_time'] < self.filtered_data['open_price'] - self.tolerance), 'open_price'
        ] = self.filtered_data['close_price_at_open_time']

        self.filtered_data.loc[
            (self.filtered_data['close_price_at_close_time'] > self.filtered_data['close_price'] + self.tolerance) |
            (self.filtered_data['close_price_at_close_time'] < self.filtered_data['close_price'] - self.tolerance), 'close_price'
        ] = self.filtered_data['close_price_at_close_time']

    def recalculate_pnl(self):
        self.filtered_data['gross_pnl'] = self.filtered_data.apply(
            lambda x: (x['close_price'] - x['open_price']) * x['base_amount'] if x['side'] == 'LONG' else
                      (x['open_price'] - x['close_price']) * x['base_amount'], axis=1)

        self.filtered_data['net_pnl'] = self.filtered_data['gross_pnl'] - self.filtered_data['fee_transaction']

    def update_combined_data(self):
        columns_to_update = ['close_price', 'open_price', 'gross_pnl', 'net_pnl']
        for column in columns_to_update:
            self.data.loc[self.data['id'].isin(self.filtered_data['id']), column] = self.data['id'].map(self.filtered_data.set_index('id')[column])

    def save_updated_data(self):
        output_file_path = os.path.join(self.base_dir, 'combined_position_data_updated.csv')
        self.data.to_csv(output_file_path, index=False)

    def run_fix(self):
        self.load_data()
        self.filter_data()
        self.load_candlestick_data()
        self.merge_price_data()
        self.adjust_prices()
        self.recalculate_pnl()
        self.update_combined_data()
        self.save_updated_data()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fixer = PositionDataFixer(base_dir)
    fixer.run_fix()
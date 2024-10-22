import os
import pandas as pd
import numpy as np
from datetime import timedelta

class PriceChangeProcessor:
    def __init__(self, file_path, output_path):
        self.file_path = file_path
        self.output_path = output_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.data['Gmt time'] = pd.to_datetime(self.data['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
        self.data.set_index('Gmt time', inplace=True)

    def calculate_price_changes(self):
        self.data['Close_Change'] = self.data['Close'].diff()
        self.data.iloc[0, self.data.columns.get_loc('Close_Change')] = self.data.iloc[0]['Close'] - self.data.iloc[0]['Open']
        self.data['Up/Down'] = self.data['Close_Change'].apply(lambda x: 'Up' if x > 0 else ('Down' if x < 0 else 'No Change'))

    def save_data(self):
        self.data.to_csv(self.output_path)

    def process(self):
        self.load_data()
        self.calculate_price_changes()
        self.save_data()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Process EURUSD file
    eurusd_file_path = os.path.join(base_dir, 'EURUSD_Candlestick_1_Hour_ASK_30.04.2020-31.12.2023.csv')
    eurusd_output_path = os.path.join(base_dir, 'EURUSD_hourly_with_changes.csv')
    eurusd_processor = PriceChangeProcessor(eurusd_file_path, eurusd_output_path)
    eurusd_processor.process()

    # Process GBPUSD file
    gbpusd_file_path = os.path.join(base_dir, 'GBPUSD_Candlestick_1_Hour_ASK_30.04.2020-31.12.2023.csv')
    gbpusd_output_path = os.path.join(base_dir, 'GBPUSD_hourly_with_changes.csv')
    gbpusd_processor = PriceChangeProcessor(gbpusd_file_path, gbpusd_output_path)
    gbpusd_processor.process()
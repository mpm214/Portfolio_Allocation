import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

class FinancialMetricsProcessor:
    def __init__(self, base_dir):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f', utc=True)
        df = df.rename(columns={'Gmt time': 'Date'})
        df.set_index('Date', inplace=True)
        return df

    def calculate_sma(self, data, window):
        return data['Close'].rolling(window=window).mean()

    def calculate_ema(self, data, window):
        return data['Close'].ewm(span=window, adjust=False).mean()

    def calculate_rsi(self, data, window):
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, data, short_window, long_window, signal_window):
        short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
        long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal

    def calculate_bollinger_bands(self, data, window, num_std_dev):
        rolling_mean = data['Close'].rolling(window).mean()
        rolling_std = data['Close'].rolling(window).std()
        upper_band = rolling_mean + (rolling_std * num_std_dev)
        lower_band = rolling_mean - (rolling_std * num_std_dev)
        return rolling_mean, upper_band, lower_band

    def calculate_obv(self, data):
        data['OBV'] = 0.0
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
                data.loc[data.index[i], 'OBV'] = data['OBV'].iloc[i - 1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
                data.loc[data.index[i], 'OBV'] = data['OBV'].iloc[i - 1] - data['Volume'].iloc[i]
            else:
                data.loc[data.index[i], 'OBV'] = data['OBV'].iloc[i - 1]
        return data

    def calculate_vwap(self, data):
        data['Cumulative_Volume'] = data['Volume'].cumsum()
        data['Cumulative_VWAP'] = (data['Close'] * data['Volume']).cumsum()
        data['VWAP'] = data['Cumulative_VWAP'] / data['Cumulative_Volume']
        return data

    def stochastic_oscillator(self, data, k_window, d_window):
        data['L14'] = data['Low'].rolling(window=k_window).min()
        data['H14'] = data['High'].rolling(window=k_window).max()
        data['K_PCT'] = 100 * ((data['Close'] - data['L14']) / (data['H14'] - data['L14']))
        data['D_PCT'] = data['K_PCT'].rolling(window=d_window).mean()
        return data

    def rate_of_change(self, data, periods):
        data['ROC'] = data['Close'].pct_change(periods=periods) * 100
        return data

    def calculate_adx_psar(self, data, window, acceleration_step, maximum_acceleration):
        data['TR'] = np.maximum(data['High'] - data['Low'], np.maximum(abs(data['High'] - data['Close'].shift(1)), abs(data['Low'] - data['Close'].shift(1))))
        data['DM_up'] = np.where((data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']), data['High'] - data['High'].shift(1), 0)
        data['DM_down'] = np.where((data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)), data['Low'].shift(1) - data['Low'], 0)

        data['TR_Smooth'] = data['TR'].rolling(window=window).sum()
        data['DM_up_Smooth'] = data['DM_up'].rolling(window=window).sum()
        data['DM_down_Smooth'] = data['DM_down'].rolling(window=window).sum()

        data['DI_up'] = 100 * (data['DM_up_Smooth'] / data['TR_Smooth'])
        data['DI_down'] = 100 * (data['DM_down_Smooth'] / data['TR_Smooth'])

        data['DX'] = 100 * (abs(data['DI_up'] - data['DI_down']) / abs(data['DI_up'] + data['DI_down']))
        data['ADX'] = data['DX'].rolling(window=window).mean()

        data['PSAR'] = data['Close'].iloc[0]
        data['PSAR_direction'] = 1
        return data

    def calculate_cmf(self, data, window):
        data['MFM'] = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        data['MFM'].replace([np.inf, -np.inf], 0, inplace=True)
        data['MFM'].fillna(0, inplace=True)
        data['MFV'] = data['MFM'] * data['Volume']
        data['CMF'] = data['MFV'].rolling(window=window).sum() / data['Volume'].rolling(window=window).sum()
        data['CMF'].replace([np.inf, -np.inf], 0, inplace=True)
        data['CMF'].fillna(0, inplace=True)
        return data

    def process_file(self, input_file, output_file):
        df = self.load_data(input_file)
        windows = [10*24, 50*24, 200*24]

        steps = 13  
        with tqdm(total=steps, desc="Processing Financial Metrics", unit="step") as pbar:
            for window in windows:
                df[f'SMA_{window}'] = self.calculate_sma(df, window)
                df[f'EMA_{window}'] = self.calculate_ema(df, window)
            pbar.update(1)

            df['RSI'] = self.calculate_rsi(df, 14 * 24)
            pbar.update(1)

            df['MACD'], df['Signal'] = self.calculate_macd(df, 12 * 24, 26 * 24, 9 * 24)
            pbar.update(1)

            df['Rolling_Mean'], df['Upper_Band'], df['Lower_Band'] = self.calculate_bollinger_bands(df, 20 * 24, 3)
            pbar.update(1)

            df = self.calculate_obv(df)
            pbar.update(1)

            df = self.calculate_vwap(df)
            pbar.update(1)

            df = self.stochastic_oscillator(df, 14 * 24, 3 * 24)
            pbar.update(1)

            df = self.rate_of_change(df, 3 * 24)
            pbar.update(1)

            df = self.calculate_adx_psar(df, 14 * 24, 0.01, 0.1)
            pbar.update(1)

            df = self.calculate_cmf(df, 20 * 24)
            pbar.update(1)

            # MACD crossovers
            df['Crossover'] = df['MACD'] - df['Signal']
            df['CrossUp'] = (df['Crossover'] > 0) & (df['Crossover'].shift(1) <= 0)
            df['CrossDown'] = (df['Crossover'] < 0) & (df['Crossover'].shift(1) >= 0)
            df['CrossUp'] = df['CrossUp'].astype(int)
            df['CrossDown'] = df['CrossDown'].astype(int)
            pbar.update(1)

            # Bollinger Band-based features
            df['Band_Width'] = df['Upper_Band'] - df['Lower_Band']
            squeeze_threshold = df['Band_Width'].mean() * 0.5
            df['Squeeze'] = df['Band_Width'] < squeeze_threshold
            df['Squeeze'] = df['Squeeze'].astype(int)

            df['Breakout'] = (df['Close'] > df['Upper_Band']) | (df['Close'] < df['Lower_Band'])
            df['Breakout'] = df['Breakout'].astype(int)

            df['Reversal_Up'] = (df['Close'] > df['Upper_Band']) & (df['Close'].shift(1) <= df['Upper_Band'])
            df['Reversal_Down'] = (df['Close'] < df['Lower_Band']) & (df['Close'].shift(1) >= df['Lower_Band'])
            df['Reversal_Up'] = df['Reversal_Up'].astype(int)
            df['Reversal_Down'] = df['Reversal_Down'].astype(int)

            extreme_volatility_threshold = df['Band_Width'].mean() + (df['Band_Width'].std() * 2)
            df['Extreme_Volatility'] = df['Band_Width'] > extreme_volatility_threshold
            df['Extreme_Volatility'] = df['Extreme_Volatility'].astype(int)
            pbar.update(1)

            df = df.loc['2021-06-01 00:00':'2023-12-31 23:00']
            df.to_csv(output_file)
            pbar.update(1)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processor = FinancialMetricsProcessor(base_dir)

    file_path_1 = os.path.join(base_dir, 'EURUSD_Candlestick_1_Hour_ASK_30.04.2020-31.12.2023.csv')
    output_path_1 = os.path.join(base_dir, 'EURUSD_underlying_metrics.csv')

    file_path_2 = os.path.join(base_dir, 'GBPUSD_Candlestick_1_Hour_ASK_30.04.2020-31.12.2023.csv')
    output_path_2 = os.path.join(base_dir, 'GBPUSD_underlying_metrics.csv')

    processor.process_file(file_path_1, output_path_1)
    processor.process_file(file_path_2, output_path_2)

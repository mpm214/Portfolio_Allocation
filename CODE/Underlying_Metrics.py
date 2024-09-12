import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def process_file(file_path, output_path):
    # Load the provided CSV file
    df = pd.read_csv(file_path)

    # Convert the 'Gmt time' column to datetime
    df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')

    # Set the 'Gmt time' column as the index
    df.set_index('Gmt time', inplace=True)

    # Define a function to calculate SMA
    def calculate_sma(data, window):
        return data['Close'].rolling(window=window).mean()

    # Define a function to calculate EMA
    def calculate_ema(data, window):
        return data['Close'].ewm(span=window, adjust=False).mean()

    # Define a function to calculate RSI
    def calculate_rsi(data, window):
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Define a function to calculate MACD
    def calculate_macd(data, short_window, long_window, signal_window):
        short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
        long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal

    # Define a function to calculate Bollinger Bands
    def calculate_bollinger_bands(data, window, num_std_dev):
        rolling_mean = data['Close'].rolling(window).mean()
        rolling_std = data['Close'].rolling(window).std()
        upper_band = rolling_mean + (rolling_std * num_std_dev)
        lower_band = rolling_mean - (rolling_std * num_std_dev)
        return rolling_mean, upper_band, lower_band

    # Define the function to calculate ADX and Parabolic SAR
    def calculate_adx_psar(data, window, initial_acceleration, acceleration_step, maximum_acceleration):
        data['TR'] = np.maximum(data['High'] - data['Low'], np.maximum(abs(data['High'] - data['Close'].shift(1)), abs(data['Low'] - data['Close'].shift(1))))
        data['DM_up'] = np.where((data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']), data['High'] - data['High'].shift(1), 0)
        data['DM_up'] = np.where(data['DM_up'] < 0, 0, data['DM_up'])
        data['DM_down'] = np.where((data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)), data['Low'].shift(1) - data['Low'], 0)
        data['DM_down'] = np.where(data['DM_down'] < 0, 0, data['DM_down'])

        data['TR_Smooth'] = data['TR'].rolling(window=window).sum()
        data['DM_up_Smooth'] = data['DM_up'].rolling(window=window).sum()
        data['DM_down_Smooth'] = data['DM_down'].rolling(window=window).sum()

        data['DI_up'] = 100 * (data['DM_up_Smooth'] / data['TR_Smooth'])
        data['DI_down'] = 100 * (data['DM_down_Smooth'] / data['TR_Smooth'])

        data['DX'] = 100 * (abs(data['DI_up'] - data['DI_down']) / abs(data['DI_up'] + data['DI_down']))
        data['ADX'] = data['DX'].rolling(window=window).mean()

        data['PSAR'] = data['Close'].iloc[0]
        data['PSAR_direction'] = 1  # 1 for up, 0 for down

        for i in range(1, len(data)):
            previous_psar = data['PSAR'].iloc[i - 1]
            previous_high = data['High'].iloc[i - 1]
            previous_low = data['Low'].iloc[i - 1]
            current_high = data['High'].iloc[i]
            current_low = data['Low'].iloc[i]

            if data['PSAR_direction'].iloc[i - 1] == 1:  # direction is up
                data.loc[data.index[i], 'PSAR'] = previous_psar + acceleration_step * (previous_high - previous_psar)
                if current_low < data['PSAR'].iloc[i]:
                    data.loc[data.index[i], 'PSAR_direction'] = 0  # change direction to down
                    data.loc[data.index[i], 'PSAR'] = previous_high
            else:  # direction is down
                data.loc[data.index[i], 'PSAR'] = previous_psar + acceleration_step * (previous_low - previous_psar)
                if current_high > data['PSAR'].iloc[i]:
                    data.loc[data.index[i], 'PSAR_direction'] = 1  # change direction to up
                    data.loc[data.index[i], 'PSAR'] = previous_low

        return data

    # Define a function to calculate OBV
    def calculate_obv(data):
        data['OBV'] = 0.0  # Initialize OBV with float type
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
                data.loc[data.index[i], 'OBV'] = data['OBV'].iloc[i - 1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
                data.loc[data.index[i], 'OBV'] = data['OBV'].iloc[i - 1] - data['Volume'].iloc[i]
            else:
                data.loc[data.index[i], 'OBV'] = data['OBV'].iloc[i - 1]
        return data

    # Define a function to calculate CMF
    def calculate_cmf(data, window):
        # Money Flow Multiplier (MFM)
        data['MFM'] = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        # Handle division by zero in MFM calculation
        data['MFM'].replace([np.inf, -np.inf], 0, inplace=True)
        data['MFM'].fillna(0, inplace=True)
        
        # Money Flow Volume (MFV)
        data['MFV'] = data['MFM'] * data['Volume']
        
        # Chaikin Money Flow (CMF)
        data['CMF'] = data['MFV'].rolling(window=window).sum() / data['Volume'].rolling(window=window).sum()
        
        # Handle potential issues with rolling sums
        data['CMF'].replace([np.inf, -np.inf], 0, inplace=True)
        data['CMF'].fillna(0, inplace=True)
        
        return data

    # Define a function to calculate VWAP
    def calculate_vwap(data):
        data['Cumulative_Volume'] = data['Volume'].cumsum()
        data['Cumulative_VWAP'] = (data['Close'] * data['Volume']).cumsum()
        data['VWAP'] = data['Cumulative_VWAP'] / data['Cumulative_Volume']
        return data

    # Define a function to calculate Stochastic Oscillator
    def stochastic_oscillator(data, k_window, d_window):
        data['L14'] = data['Low'].rolling(window=k_window).min()
        data['H14'] = data['High'].rolling(window=k_window).max()
        data['K_PCT'] = 100 * ((data['Close'] - data['L14']) / (data['H14'] - data['L14']))
        data['D_PCT'] = data['K_PCT'].rolling(window=d_window).mean()
        return data

    # Define a function to calculate Rate of Change (ROC)
    def rate_of_change(data, periods):
        data['ROC'] = data['Close'].pct_change(periods=periods) * 100
        return data
    
    def identify_levels(df, prominence=1):
        peaks, _ = find_peaks(df['Close'], prominence=prominence)
        troughs, _ = find_peaks(-df['Close'], prominence=prominence)
        return peaks, troughs

    # Define different window periods and other parameters
    windows = [10*24, 50*24, 200*24]  # 10 days, 50 days, 200 days worth of hours
    window_rsi = 14 * 24  # 14 days worth of hours
    short_window = 12 * 24  # 12 days worth of hours
    long_window = 26 * 24   # 26 days worth of hours
    signal_window = 9 * 24  # 9 days worth of hours
    bb_window = 20 * 24  # 20 days worth of hours
    num_std_dev = 3
    adx_psar_window = 14 * 24  # 14 days worth of hours
    cmf_window = 20*24  # 20 periods for CMF calculation
    stochastic_k_window = 14*24  # Typical value: 14
    stochastic_d_window = 3*24   # Typical value: 3
    roc_periods = 3*24          # Typical value: 12
    initial_acceleration = 0.01
    acceleration_step = 0.01
    maximum_acceleration = 0.1
    prominence_level = 0.005

    # Calculate and add SMAs and EMAs to the DataFrame for each window period
    for window in windows:
        df[f'SMA_{window}'] = calculate_sma(df, window)
        df[f'EMA_{window}'] = calculate_ema(df, window)

    # Calculate and add RSI to the DataFrame
    df['RSI'] = calculate_rsi(df, window_rsi)

    # Calculate and add MACD to the DataFrame
    df['MACD'], df['Signal'] = calculate_macd(df, short_window, long_window, signal_window)

    # Identify crossovers
    df['Crossover'] = df['MACD'] - df['Signal']
    df['CrossUp'] = (df['Crossover'] > 0) & (df['Crossover'].shift(1) <= 0)
    df['CrossDown'] = (df['Crossover'] < 0) & (df['Crossover'].shift(1) >= 0)
    df['CrossUp'] = df['CrossUp'].astype(int)
    df['CrossDown'] = df['CrossDown'].astype(int)

    # Calculate Bollinger Bands and add to the DataFrame
    df['Rolling_Mean'], df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(df, bb_window, num_std_dev)

    # Identify Squeezes (when the band width is narrow)
    df['Band_Width'] = df['Upper_Band'] - df['Lower_Band']
    squeeze_threshold = df['Band_Width'].mean() * 0.5  # Squeeze threshold as 50% of mean band width
    df['Squeeze'] = df['Band_Width'] < squeeze_threshold
    df['Squeeze'] = df['Squeeze'].astype(int)

    # Identify Breakouts (when price moves outside the bands)
    df['Breakout'] = (df['Close'] > df['Upper_Band']) | (df['Close'] < df['Lower_Band'])
    df['Breakout'] = df['Breakout'].astype(int)

    # Identify Reversals (when price touches upper or lower band and then moves towards the middle)
    df['Reversal_Up'] = (df['Close'] > df['Upper_Band']) & (df['Close'].shift(1) <= df['Upper_Band'])
    df['Reversal_Down'] = (df['Close'] < df['Lower_Band']) & (df['Close'].shift(1) >= df['Lower_Band'])
    df['Reversal_Up'] = df['Reversal_Up'].astype(int)
    df['Reversal_Down'] = df['Reversal_Down'].astype(int)

    # Identify Extreme Volatility (when the band width is wide)
    extreme_volatility_threshold = df['Band_Width'].mean() + (df['Band_Width'].std()*2)
    df['Extreme_Volatility'] = df['Band_Width'] > extreme_volatility_threshold
    df['Extreme_Volatility'] = df['Extreme_Volatility'].astype(int)

    # Calculate and add ADX and Parabolic SAR to the DataFrame
    df = calculate_adx_psar(df, adx_psar_window, initial_acceleration, acceleration_step, maximum_acceleration)

    # Calculate and add OBV, CMF, and VWAP to the DataFrame
    df = calculate_obv(df)
    df = calculate_cmf(df, cmf_window)
    df = calculate_vwap(df)

    # Calculate and add Stochastic Oscillator and ROC to the DataFrame
    df = stochastic_oscillator(df, stochastic_k_window, stochastic_d_window)
    df = rate_of_change(df, roc_periods)
    
    # Support and Resistence Levels
    peaks, troughs = identify_levels(df, prominence=prominence_level)
    df['resistance_levels'] = df['Close'].iloc[peaks]
    df['support_levels'] = df['Close'].iloc[troughs]
    df['resistance_levels'].fillna(0, inplace=True)
    df['support_levels'].fillna(0, inplace=True)

    df = df.loc['2021-06-01 00:00':'2023-12-31 23:00']
    # Save the revised DataFrame to a new CSV file
    df.to_csv(output_path)

# Get the base directory from the environment variable
base_dir = os.environ.get('DYNAMIC_PORTFOLIO_ALLOCATION', os.getcwd())

# File paths for the two different files
file_path_1 = os.path.join(base_dir, 'Data_Prep', 'EURUSD_Candlestick_1_Hour_ASK_30.04.2020-31.12.2023.csv')
output_path_1 = os.path.join(base_dir, 'Data_Prep', 'EURUSD_Underlying_Hour_metrics.csv')

file_path_2 = os.path.join(base_dir, 'Data_Prep', 'GBPUSD_Candlestick_1_Hour_ASK_30.04.2020-31.12.2023.csv')
output_path_2 = os.path.join(base_dir, 'Data_Prep', 'GBPUSD_Underlying_Hour_metrics.csv')

# Process each file and save the revised DataFrame
process_file(file_path_1, output_path_1)
process_file(file_path_2, output_path_2)

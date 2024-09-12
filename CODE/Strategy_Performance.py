import pandas as pd
import numpy as np
import os
from scipy.stats import linregress

# Get the base directory from the environment variable
base_dir = os.environ.get('DYNAMIC_PORTFOLIO_ALLOCATION', os.getcwd())

# Define file paths
pnl_path = os.path.join(base_dir, 'Data_Prep', 'combined_daily_pnl_data_updated.csv')
positions_path = os.path.join(base_dir, 'Data_Prep', 'combined_position_data_updated.csv')

# Load data
pnl_data = pd.read_csv(pnl_path)
positions_data = pd.read_csv(positions_path)

# Ensure the Date column is in datetime format
pnl_data['Date'] = pd.to_datetime(pnl_data['Date'])

# Calculate the rolling 30-day standard deviation for each unique strategy_name
pnl_data['rolling_std_30'] = pnl_data.groupby('strategy_name')['net_pnl_change'].rolling(window=30).std().reset_index(level=0, drop=True)
pnl_data['rolling_std_30'] = pnl_data.groupby('strategy_name')['rolling_std_30'].bfill()

# Calculate the first (rate of change) and second derivative (momentum/speed) of the rolling_std_30
pnl_data['std_ROC'] = pnl_data.groupby('strategy_name')['rolling_std_30'].transform(lambda x: x.diff())
pnl_data['std_ROC'] = pnl_data.groupby('strategy_name')['std_ROC'].bfill()
pnl_data['std_Momentum'] = pnl_data.groupby('strategy_name')['std_ROC'].transform(lambda x: x.diff())
pnl_data['std_Momentum'] = pnl_data.groupby('strategy_name')['std_Momentum'].bfill()

# Calculate the rolling 30-day average return for each unique strategy_name
pnl_data['rolling_avg_30'] = pnl_data.groupby('strategy_name')['net_pnl_change'].rolling(window=30).mean().reset_index(level=0, drop=True)
pnl_data['rolling_avg_30'] = pnl_data.groupby('strategy_name')['rolling_avg_30'].bfill()

# Calculate the first (rate of change) and second derivative (momentum/speed) of the rolling_avg_30
pnl_data['PnL_Avg_ROC'] = pnl_data.groupby('strategy_name')['rolling_avg_30'].transform(lambda x: x.diff())
pnl_data['PnL_Avg_ROC'] = pnl_data.groupby('strategy_name')['PnL_Avg_ROC'].bfill()
pnl_data['PnL_Avg_Momentum'] = pnl_data.groupby('strategy_name')['PnL_Avg_ROC'].transform(lambda x: x.diff())
pnl_data['PnL_Avg_Momentum'] = pnl_data.groupby('strategy_name')['PnL_Avg_Momentum'].bfill()

# Calculate the Sharpe Ratio without risk-free rate
pnl_data['sharpe_ratio'] = pnl_data['rolling_avg_30'] / pnl_data['rolling_std_30']
pnl_data['sharpe_ratio'] = pnl_data.groupby('strategy_name')['sharpe_ratio'].bfill()
pnl_data.replace({'sharpe_ratio': [np.inf, -np.inf]}, 0, inplace=True)

# Calculate the first (rate of change) and second derivative (momentum/speed) of the sharpe_ratio
pnl_data['Sharpe_ROC'] = pnl_data.groupby('strategy_name')['sharpe_ratio'].transform(lambda x: x.diff())
pnl_data['Sharpe_ROC'] = pnl_data.groupby('strategy_name')['Sharpe_ROC'].bfill()
pnl_data['Sharpe_Momentum'] = pnl_data.groupby('strategy_name')['Sharpe_ROC'].transform(lambda x: x.diff())
pnl_data['Sharpe_Momentum'] = pnl_data.groupby('strategy_name')['Sharpe_Momentum'].bfill()

# Calculate Peak
pnl_data['rolling_Peak_30'] = pnl_data.groupby('strategy_name')['net_pnl'].rolling(window=30).max().reset_index(level=0, drop=True)
pnl_data['rolling_Peak_30'] = pnl_data.groupby('strategy_name')['rolling_Peak_30'].bfill()

# Calculate Drawdown
pnl_data['Drawdown'] = (pnl_data['net_pnl'] - pnl_data['rolling_Peak_30']) / pnl_data['rolling_Peak_30']
pnl_data['Drawdown'] = pnl_data.groupby('strategy_name')['Drawdown'].bfill()
pnl_data.replace({'Drawdown': [np.inf, -np.inf]}, 0, inplace=True)

# Add Long/Short column
pnl_data['Long_Short'] = pnl_data['strategy_name'].apply(lambda x: 1 if x.startswith('L_') else -1 if x.startswith('S_') else 0)

# Recovery Time
pnl_data['rolling_Peak_Recovery_Days'] = pnl_data.apply(lambda row: 1 if row['rolling_Peak_30'] == row['net_pnl'] else 0, axis=1)

def calculate_recovery_time(series):
    recovery_times = []
    count = 0
    for value in series:
        if value == 1:
            recovery_times.append(count)
            count = 0
        else:
            recovery_times.append(0)
            count += 1
    return recovery_times

pnl_data['Recovery_Time'] = pnl_data.groupby('strategy_name')['rolling_Peak_Recovery_Days'].transform(calculate_recovery_time)
pnl_data['Rolling_Max_Recovery_Time'] = pnl_data.groupby('strategy_name')['Recovery_Time'].rolling(window=30).max().reset_index(level=0, drop=True)
pnl_data['Rolling_Max_Recovery_Time'] = pnl_data.groupby('strategy_name')['Rolling_Max_Recovery_Time'].bfill()

# Calculate the 30-day slope of net_pnl
pnl_data['PnL_slope_30'] = pnl_data.groupby('strategy_name')['net_pnl'].transform(lambda x: (x - x.shift(29)) / 30)
pnl_data['PnL_slope_30'] = pnl_data.groupby('strategy_name')['PnL_slope_30'].bfill()

# Calculate the second derivative (momentum) of the 30-day slope
pnl_data['PnL_momentum_30'] = pnl_data.groupby('strategy_name')['PnL_slope_30'].transform(lambda x: x.diff())
pnl_data['PnL_momentum_30'] = pnl_data.groupby('strategy_name')['PnL_momentum_30'].bfill()

# Time Based Features
pnl_data['month'] = pnl_data['Date'].dt.month
pnl_data['day_of_week'] = pnl_data['Date'].dt.dayofweek
pnl_data['day_of_month'] = pnl_data['Date'].dt.day

# Output to CSV
# Reformat the Date column to YYYY-MM-DD
pnl_data['Date'] = pd.to_datetime(pnl_data['Date']).dt.date
output_path = os.path.join(base_dir, 'Feature_Engineering', 'Strategy_Metrics.csv')
pnl_data.to_csv(output_path, index=False)
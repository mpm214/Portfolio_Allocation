import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# Get the base directory from the environment variable
base_dir = os.environ.get('DYNAMIC_PORTFOLIO_ALLOCATION', os.getcwd())

# Define the file path
file_path = os.path.join(base_dir, 'Data_Prep', 'combined_daily_pnl_data_updated.csv')

# Load the CSV file
df = pd.read_csv(file_path)

# Convert the "Date" column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Function to calculate net pnl change based on a given frequency (monthly)
def calculate_net_pnl_change(df):
    result_df = df.groupby(['strategy_name', df['Date'].dt.to_period('M')])['net_pnl_change'].sum().reset_index()
    result_df.rename(columns={'Date': 'Period'}, inplace=True)
    return result_df

# Function to calculate top 10 strategies based on monthly pnl change
def calculate_top_10(df):
    result_df = calculate_net_pnl_change(df)
    top_10_results = pd.DataFrame()

    # Loop through each unique period and select top 10
    for period in result_df['Period'].unique():
        period_data = result_df[result_df['Period'] == period]
        top_10_period = period_data.nlargest(10, 'net_pnl_change')
        top_10_results = pd.concat([top_10_results, top_10_period])

    return top_10_results

# Directory to save the CSV file
output_dir = os.path.join(base_dir, 'Top_N_Results')
os.makedirs(output_dir, exist_ok=True)

# Calculate top 10 strategies for monthly frequency and save to CSV
top_10_results = calculate_top_10(df)
output_path = os.path.join(output_dir, "top_10_M.csv")
top_10_results.to_csv(output_path, index=False)

# Load the existing daily data
input_path = os.path.join(base_dir, 'Feature_Engineering', 'Strategy_Metrics.csv')
data = pd.read_csv(input_path, parse_dates=['Date'], index_col='Date')
data = data[data.index <= '2023-12-31']

# Exclude period "2023-12"
df_ranked = top_10_results[top_10_results['Period'] < '2023-12']

# Rank strategies within each period
df_ranked['Rank'] = df_ranked.groupby('Period')['net_pnl_change'].rank(method='first', ascending=True).astype(int)

# Function to extrapolate monthly data to daily
def extrapolate_to_daily(df):
    daily_df = pd.DataFrame()

    for _, row in df.iterrows():
        start_date = row['Period'].to_timestamp()
        end_date = start_date + pd.offsets.MonthEnd(0)
        daily_dates = pd.date_range(start_date, end_date, freq='D')
        
        temp_df = pd.DataFrame({
            'Date': daily_dates,
            'strategy_name': row['strategy_name'],
            'Rank': row['Rank']
        })
        
        daily_df = pd.concat([daily_df, temp_df], ignore_index=True)
    
    return daily_df

# Function to shift the periods back by one month
def shift_periods(df):
    df['Period'] = pd.to_datetime(df['Period']) - pd.DateOffset(months=1)
    df['Period'] = df['Period'].dt.to_period('M')
    return df

# Shift periods back by one month
df_ranked_shifted = shift_periods(df_ranked)

# Extrapolate the ranked monthly data to daily
daily_df_shift_back = extrapolate_to_daily(df_ranked_shifted)

# Merge with selected strategies
df_ranked_shifted.set_index('Date', inplace=True)
selected_strategies_with_metrics = pd.merge(data, df_ranked_shifted, on=['Date', 'strategy_name'], how='left')

# Fill NaN values in Rank column with 0
selected_strategies_with_metrics['Rank'].fillna(0, inplace=True)

# Define the output path
output_path = os.path.join(base_dir, 'Strategy_Metrics_Selected.csv')
selected_strategies_with_metrics.to_csv(output_path)

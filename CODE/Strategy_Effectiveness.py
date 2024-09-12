import pandas as pd
import os
from datetime import timedelta

def process_trades(file_path):
    """Process trade data to mark open/close intervals."""
    df = pd.read_csv(file_path)
    df[['open_time', 'close_time']] = df[['open_time', 'close_time']].apply(pd.to_datetime)
    hourly_index = pd.date_range('2021-06-01', '2024-01-01', freq='h')

    result_dfs = []
    for strategy in df['strategy_name'].unique():
        strategy_df = df[df['strategy_name'] == strategy]
        strategy_result_df = pd.DataFrame(index=hourly_index)
        strategy_result_df.index.name = 'Time'
        strategy_result_df[['Trade Status', 'Trade Open/Close']] = 'No Trade'

        for _, trade in strategy_df.iterrows():
            open_hour, close_hour = trade[['open_time', 'close_time']].dt.floor('h')
            strategy_result_df.loc[(strategy_result_df.index >= open_hour) & (strategy_result_df.index <= close_hour), 'Trade Status'] = 'Open'
            strategy_result_df.loc[open_hour, 'Trade Open/Close'] = 'Open Trade'
            strategy_result_df.loc[close_hour, 'Trade Open/Close'] = 'Close Trade'

        strategy_result_df['Strategy'] = strategy
        result_dfs.append(strategy_result_df)

    return pd.concat(result_dfs).reset_index()

def load_and_filter(file_path, time_column, start_time, end_time):
    """Load and filter data by the specified time range."""
    df = pd.read_csv(file_path)
    df[time_column] = pd.to_datetime(df[time_column])
    return df[(df[time_column] >= start_time) & (df[time_column] <= end_time)]

def calculate_price_changes(price_data):
    """Calculate price changes."""
    price_data = price_data.sort_values(by='Gmt time')
    price_data['Close_Change'] = price_data['Close'].diff().shift(-1)
    return price_data

def calculate_ratios(data, up_value, down_value, is_long):
    """Calculate precision, downside ratio, combined ratio, and magnitude ratios."""
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

def calculate_strategy_ratios(trades_df, eurusd_df, gbpusd_df, periods):
    """Calculate strategy ratios for each unique strategy."""
    results = []
    eurusd_df, gbpusd_df = map(calculate_price_changes, [eurusd_df, gbpusd_df])

    for strategy, period_slices, asset_data in zip(
            trades_df['Strategy'].unique(),
            [slice_data_by_period(trades_df.merge(eurusd_df, on='Time'), periods, 'Gmt time') if "EURUSD" in strategy else slice_data_by_period(trades_df.merge(gbpusd_df, on='Time'), periods, 'Gmt time')],
            [eurusd_df, gbpusd_df]
    ):
        is_long = strategy.startswith('L')
        for train_data, test_data in period_slices:
            strategy_test_data = test_data[test_data['Strategy'] == strategy]
            ratios = calculate_ratios(strategy_test_data, 'Up', 'Down', is_long)
            results.append({
                'Strategy': strategy,
                'Period': f"{train_data['Gmt time'].min().date()} to {test_data['Gmt time'].max().date()}",
                'Precision': ratios[0],
                'Downside_Ratio': ratios[1],
                'Combined_Ratio': ratios[2],
                'Precision_Magnitude': ratios[3],
                'Downside_Ratio_Magnitude': ratios[4],
                'Combined_Ratio_Magnitude': ratios[5]
            })
    return pd.DataFrame(results)

def main():
    base_dir = os.environ.get('DYNAMIC_PORTFOLIO_ALLOCATION', os.getcwd())
    file_paths = {
        'trades': os.path.join(base_dir, 'Feature_Engineering', 'open_trades_times_by_strategy.csv'),
        'eurusd': os.path.join(base_dir, 'Feature_Engineering', 'EURUSD_hourly_with_changes.csv'),
        'gbpusd': os.path.join(base_dir, 'Feature_Engineering', 'GBPUSD_hourly_with_changes.csv')
    }
    start_time, end_time = pd.Timestamp('2021-06-01 00:00:00'), pd.Timestamp('2024-01-01 23:00:00')

    periods = generate_periods(start_time, end_time)
    trades_df, eurusd_df, gbpusd_df = map(lambda p: load_and_filter(p[0], p[1], start_time, end_time), [
        (file_paths['trades'], 'Time'),
        (file_paths['eurusd'], 'Gmt time'),
        (file_paths['gbpusd'], 'Gmt time')
    ])

    ratios_df = calculate_strategy_ratios(trades_df, eurusd_df, gbpusd_df, periods)
    output_path = os.path.join(base_dir, 'Feature_Engineering', 'strategy_ratios.csv')
    ratios_df.to_csv(output_path, index=False)
    print("Ratios calculated and saved to:", output_path)

if __name__ == "__main__":
    main()

# Post-process the output
ratios_df = pd.read_csv(output_path).fillna(0)
ratios_df.to_csv(output_path, index=False)

# Expand the strategy_ratios data to daily
strategy_ratios = pd.read_csv(input_path)
strategy_ratios['Period'] = pd.to_datetime(strategy_ratios['Period'].str.split(' to ').str[1])

def fix_date(date_str):
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except Exception:
        return pd.NaT

strategy_ratios['Period'] = strategy_ratios['Period'].apply(fix_date).dropna()

all_expanded_df = pd.concat([
    pd.DataFrame(index=pd.date_range('2021-06-01', group.index.max(), freq='D')).merge(
        group.set_index('Period'), how='left', left_index=True, right_index=True
    ).fillna(method='bfill').assign(Strategy=strategy)
    for strategy, group in strategy_ratios.groupby('Strategy')
]).reset_index().rename(columns={'index': 'Date'})

all_expanded_df.to_csv(output_path, index=False)

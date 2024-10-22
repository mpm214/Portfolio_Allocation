import os
import pandas as pd

def select_bottom_10_each_month_to_csv(df_allocation, output_file, num_strategies=10):
    """Selects the bottom 10 strategies with the lowest total net_pnl_change for each month/year and writes the results to a CSV file."""
    df_allocation = pd.read_csv(allocation_file)
    df_allocation['Date'] = pd.to_datetime(df_allocation['Date'], utc=True)
    df_allocation['YearMonth'] = df_allocation['Date'].dt.to_period('M')
    cumulative_bottom_totals = []
    periods = []
    results = []
    cumulative_bottom_sum = 0
    
    for period, df_month in df_allocation.groupby('YearMonth'):
        strategy_totals = df_month.groupby('strategy_name')['net_pnl_change'].sum()
        bottom_10_strategies = strategy_totals.nsmallest(num_strategies)
        grand_total_bottom = bottom_10_strategies.sum() / num_strategies
        cumulative_bottom_sum += grand_total_bottom
        cumulative_bottom_totals.append(cumulative_bottom_sum)
        periods.append(period.start_time)
        
        for strategy, pnl in bottom_10_strategies.items():
            results.append({'Period': period, 'Strategy': strategy, 'net_pnl_change': pnl, 'Grand_Total_Bottom_10_Divided_by_10': grand_total_bottom})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    return pd.Series(cumulative_bottom_totals, index=periods)

def mark_bottom_10_strategies(allocation_file, bottom_10_file, output_file):
    """Marks the bottom 10 strategies with a 1 for each hourly time row in the allocation file."""
    df_allocation = pd.read_csv(allocation_file)
    df_bottom_10 = pd.read_csv(bottom_10_file)
    df_allocation['Date'] = pd.to_datetime(df_allocation['Date'], utc=True)
    df_allocation['Period'] = df_allocation['Date'].dt.to_period('M')
    df_bottom_10['Period'] = pd.to_datetime(df_bottom_10['Period']).dt.to_period('M')
    df_allocation['bottom_10'] = 0
    df_merged = df_allocation.merge(df_bottom_10[['Period', 'Strategy']], 
                                    how='left', 
                                    left_on=['YearMonth', 'strategy_name'], 
                                    right_on=['Period', 'Strategy'])

    df_allocation.loc[df_merged['Strategy'].notnull(), 'bottom_10'] = 1
    df_allocation.drop(columns=['YearMonth'], inplace=True)
    df_allocation.to_csv(output_file, index=False)
    print(f"Updated allocation file saved to {output_file}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    allocation_file = os.path.join(base_dir, 'Allocation.csv')
    bottom_10_file = os.path.join(base_dir, 'bottom_10_strategies_by_month.csv')
    output_file = os.path.join(base_dir, 'Allocation_with_bottom_10.csv')
    cumulative_bottom = select_bottom_10_each_month_to_csv(allocation_file, output_file, num_strategies=10)
    mark_bottom_10_strategies(allocation_file, bottom_10_file, output_file)

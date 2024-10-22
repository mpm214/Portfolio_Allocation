import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def calculate_equal_weighted_benchmark(df_allocation):
    """Calculates the equal-weighted portfolio benchmark and returns cumulative net pnl."""
    df_allocation['Date'] = pd.to_datetime(df_allocation['Date'], utc=True)
    df_allocation = df_allocation.sort_values(by='Date')
    df_daily_pnl = df_allocation.groupby('Date')['net_pnl_change'].sum()  
    cumulative_equal_weighted_net_pnl = (df_daily_pnl / 167).cumsum()  # Given: 167 strategies
    return cumulative_equal_weighted_net_pnl

def monte_carlo_simulation_with_rebalancing(df_allocation, strategies_list, num_simulations, num_strategies):
    """Performs a Monte Carlo simulation with monthly rebalancing by randomly selecting strategies."""
    df_allocation['Date'] = pd.to_datetime(df_allocation['Date'], utc=True)
    df_allocation['YearMonth'] = df_allocation['Date'].dt.to_period('M')  # Group by year-month for rebalancing

    cumulative_pnl_simulations = pd.DataFrame(index=df_allocation['Date'].unique())

    for i in range(num_simulations):
        # Initialize an empty dataframe to store monthly cumulative pnl for each simulation
        monthly_cumulative_pnl = []

        for period, df_month in df_allocation.groupby('YearMonth'):
            # Randomly select 10 unique strategies for each month
            random_strategies = np.random.choice(strategies_list, size=num_strategies, replace=False)

            # Filter the allocation data for the randomly selected strategies in this month
            df_selected_strategies = df_month[df_month['strategy_name'].isin(random_strategies)]

            # Group by Date within the month and sum the net_pnl_change for the selected strategies
            df_selected_pnl = df_selected_strategies.groupby('Date')['net_pnl_change'].sum()

            # Store the cumulative sum for the month
            monthly_cumulative_pnl.append(df_selected_pnl / num_strategies)

        # Concatenate all months' pnl and calculate cumulative sum for the entire simulation
        monthly_cumulative_pnl = pd.concat(monthly_cumulative_pnl).cumsum()

        cumulative_pnl_simulations[f'simulation_{i}'] = monthly_cumulative_pnl

    return cumulative_pnl_simulations

def plot_combined_benchmarks(cumulative_pnl_simulations, cumulative_benchmark, base_dir):
    """Plots the Monte Carlo simulation and equal-weighted benchmark with a shaded region."""
    output_pdf = os.path.join(base_dir, 'combined_mc_equal_weighted_benchmark.pdf')

    # Calculate the average, minimum, and maximum cumulative PnL across all simulations
    avg_cumulative_pnl = cumulative_pnl_simulations.mean(axis=1)
    min_cumulative_pnl = cumulative_pnl_simulations.min(axis=1)
    max_cumulative_pnl = cumulative_pnl_simulations.max(axis=1)

    with PdfPages(output_pdf) as pdf:
        plt.figure(figsize=(12, 6))

        # Plot the average cumulative net PnL from the Monte Carlo simulation
        plt.plot(avg_cumulative_pnl.index, avg_cumulative_pnl, label='Monte Carlo Benchmark (10 strategies, Monthly Rebalance)', color='m')

        # Plot the equal-weighted benchmark
        plt.plot(cumulative_benchmark.index, cumulative_benchmark, label='Equal-Weighted Portfolio', color='b')

        # Shade the region between the worst and best case simulations
        plt.fill_between(avg_cumulative_pnl.index, min_cumulative_pnl, max_cumulative_pnl, color='m', alpha=0.2, label='Simulation Spread')

        plt.title('Monte Carlo Benchmark with Equal-Weighted Portfolio Overlay')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Net PnL')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print(f"Graph saved to {output_pdf}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Load the Allocation.csv file
    file_path = os.path.join(base_dir, 'Allocation.csv')
    df_allocation = pd.read_csv(file_path)

    # Step 2: Get the list of unique strategy names
    strategies_list = df_allocation['strategy_name'].unique()

    # Step 3: Perform Monte Carlo simulation with monthly rebalancing
    cumulative_pnl_simulations = monte_carlo_simulation_with_rebalancing(df_allocation, strategies_list, num_simulations=1000, num_strategies=10)

    # Step 4: Calculate the equal-weighted benchmark
    cumulative_benchmark = calculate_equal_weighted_benchmark(df_allocation)

    # Step 5: Plot the Monte Carlo benchmark and overlay the equal-weighted benchmark
    plot_combined_benchmarks(cumulative_pnl_simulations, cumulative_benchmark, base_dir)

import os
import pandas as pd
from tqdm import tqdm

class ConditionalMerger:
    def __init__(self, base_dir, strategy_file, eurusd_file, gbpusd_file, effectiveness_file):
        self.base_dir = base_dir
        self.strategy_file = strategy_file
        self.eurusd_file = eurusd_file
        self.gbpusd_file = gbpusd_file
        self.effectiveness_file = effectiveness_file

    def load_data(self):
        with tqdm(total=4, desc="Loading Data", unit="step") as pbar:
            # Load the strategy performance data
            strategy_df = pd.read_csv(self.strategy_file)
            strategy_df['Date'] = pd.to_datetime(strategy_df['Date'], utc=True)
            pbar.update(1)  # Progress after loading strategy data

            # Load the EURUSD underlying metrics
            eurusd_df = pd.read_csv(self.eurusd_file)
            eurusd_df['Date'] = pd.to_datetime(eurusd_df['Date'], utc=True)
            eurusd_df.set_index('Date', inplace=True)
            pbar.update(1)  # Progress after loading EURUSD data

            # Load the GBPUSD underlying metrics
            gbpusd_df = pd.read_csv(self.gbpusd_file)
            gbpusd_df['Date'] = pd.to_datetime(gbpusd_df['Date'], utc=True)
            gbpusd_df.set_index('Date', inplace=True)
            pbar.update(1)  # Progress after loading GBPUSD data

            # Load the strategy effectiveness data
            effectiveness_df = pd.read_csv(self.effectiveness_file)
            effectiveness_df['Date'] = pd.to_datetime(effectiveness_df['Date'], utc=True)
            pbar.update(1)  # Progress after loading strategy effectiveness data

        return strategy_df, eurusd_df, gbpusd_df, effectiveness_df

    def merge_data(self, strategy_df, eurusd_df, gbpusd_df, effectiveness_df):
        # Split the strategy data into EURUSD and GBPUSD based on CCY
        with tqdm(total=3, desc="Merging Data", unit="step") as pbar:
            eurusd_strategy = strategy_df[strategy_df['CCY'] == 'EURUSD'].copy()
            gbpusd_strategy = strategy_df[strategy_df['CCY'] == 'GBPUSD'].copy()

            # Merge the EURUSD rows with EURUSD underlying metrics
            eurusd_merged = eurusd_strategy.merge(eurusd_df, left_on='Date', right_on='Date', how='left')
            pbar.update(1)  # Progress after merging EURUSD data

            # Merge the GBPUSD rows with GBPUSD underlying metrics
            gbpusd_merged = gbpusd_strategy.merge(gbpusd_df, left_on='Date', right_on='Date', how='left')
            pbar.update(1)  # Progress after merging GBPUSD data

            # Concatenate both merged dataframes into one
            merged_strategy_df = pd.concat([eurusd_merged, gbpusd_merged], ignore_index=True)

            # Merge the strategy metrics with strategy effectiveness data
            final_df = merged_strategy_df.merge(effectiveness_df, left_on=['Date', 'strategy_name'], right_on=['Date', 'Strategy'], how='left')
            pbar.update(1)  # Progress after merging strategy effectiveness data

        return final_df

    def run(self, output_file):
        # Load data
        strategy_df, eurusd_df, gbpusd_df, effectiveness_df = self.load_data()

        # Merge data conditionally
        final_df = self.merge_data(strategy_df, eurusd_df, gbpusd_df, effectiveness_df)

        # Save the final merged DataFrame
        with tqdm(total=1, desc="Saving Data", unit="step") as pbar:
            final_df.to_csv(output_file, index=False)
            pbar.update(1)
        print(f"Merged data saved to {output_file}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    strategy_file = os.path.join(base_dir, 'strategy_hourly_performance.csv')
    eurusd_file = os.path.join(base_dir, 'EURUSD_underlying_metrics.csv')
    gbpusd_file = os.path.join(base_dir, 'GBPUSD_underlying_metrics.csv')
    effectiveness_file = os.path.join(base_dir, 'strategy_effectiveness.csv')
    output_file = os.path.join(base_dir, 'full.csv')

    merger = ConditionalMerger(base_dir, strategy_file, eurusd_file, gbpusd_file, effectiveness_file)
    merger.run(output_file)

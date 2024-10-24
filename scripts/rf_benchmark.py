import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd, DateOffset
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

class RollForwardBenchmark:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.all_strategy_ranks = None
        self.combined_hourly_pnl_data = None

    def load_data(self):
        file_path_ranks = os.path.join(self.base_dir, 'all_strategy_ranks.csv')
        self.all_strategy_ranks = pd.read_csv(file_path_ranks)
        self.all_strategy_ranks['Period'] = pd.to_datetime(self.all_strategy_ranks['Period'].astype(str), format='%Y-%m')
        self.all_strategy_ranks['Period'] = self.all_strategy_ranks['Period'] + DateOffset(months=1)

        file_path_pnl = os.path.join(self.base_dir, 'combined_hourly_pnl_data_updated.csv')
        self.combined_hourly_pnl_data = pd.read_csv(file_path_pnl)
        self.combined_hourly_pnl_data['Date'] = pd.to_datetime(self.combined_hourly_pnl_data['time'])
        self.combined_hourly_pnl_data.drop(columns=['time'], inplace=True)

    def extrapolate_to_hourly(self):
        hourly_data = pd.DataFrame()
        for _, row in tqdm(self.all_strategy_ranks.iterrows(), total=len(self.all_strategy_ranks), desc="Extrapolating to Hourly Data"):
            period = row['Period'].strftime('%Y-%m')
            start_date = pd.to_datetime(period + '-01 00:00:00+00:00')
            end_date = (start_date + MonthEnd(1)).replace(hour=23, minute=0, second=0)
            hourly_range = pd.date_range(start=start_date, end=end_date, freq='h', tz='UTC')
            strategy_data = pd.DataFrame({
                'Date': hourly_range,
                'strategy_name': row['strategy_name'],
                'Period': row['Period'],
                'Top_10_current': row['Top_10_current']
            })
            hourly_data = pd.concat([hourly_data, strategy_data])

        return hourly_data

    def merge_and_filter_top_10(self, hourly_data):
        merged_df = pd.merge(hourly_data, self.combined_hourly_pnl_data, on=['Date', 'strategy_name'], how='left')
        top_10_data = merged_df[merged_df['Top_10_current'] == 1]
        return top_10_data

    def pivot_and_calculate_cumulative(self, top_10_data):
        pivot_df = top_10_data.pivot_table(index='Date', values='net_pnl_change', aggfunc='sum').reset_index()
        pivot_df['net_pnl_change'] = pivot_df['net_pnl_change'] / 10
        pivot_df['cumulative_net_pnl'] = pivot_df['net_pnl_change'].cumsum()

        # Remove data after 2023-12-31 23:00:00
        cutoff_date = pd.to_datetime('2023-12-31 23:00:00', utc=True)
        pivot_df = pivot_df[pivot_df['Date'] <= cutoff_date]

        return pivot_df

    def plot_cumulative_net_pnl(self, pivot_df):
        # Save plot to PDF
        pdf_path = os.path.join(self.base_dir, 'rf_benchmark.pdf')
        with PdfPages(pdf_path) as pdf:
            plt.figure(figsize=(10, 6))
            sns.lineplot(x='Date', y='cumulative_net_pnl', data=pivot_df, label='Previous Month Top 10 Benchmark')
            plt.title('Cumulative Net PnL Over Time (Top 10 Strategies)')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Net PnL')
            plt.xticks(rotation=45)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    def run(self):
        self.load_data()
        hourly_data = self.extrapolate_to_hourly()
        top_10_data = self.merge_and_filter_top_10(hourly_data)
        pivot_df = self.pivot_and_calculate_cumulative(top_10_data)
        self.plot_cumulative_net_pnl(pivot_df)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    analyzer = RollForwardBenchmark(base_dir)
    analyzer.run()

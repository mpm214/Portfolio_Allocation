import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class AllocationPivot:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.df_allocation = None

    def load_allocation_data(self):
        """Loads the Allocation.csv file."""
        file_path = os.path.join(self.base_dir, 'Allocation.csv')
        self.df_allocation = pd.read_csv(file_path)
        self.df_allocation['Date'] = pd.to_datetime(self.df_allocation['Date'], utc=True)

    def pivot_by_date(self):
        """Pivots the Allocation.csv file by Date, including y_actual, y_pred_MLP, y_pred_SGD, y_pred_ABR."""
        pivot_columns = ['net_pnl_change', 'y_actual', 'y_pred_MLP', 'y_pred_SGD', 'y_pred_ABR']
        pivot_df = self.df_allocation.pivot_table(
            index='Date',
            columns='strategy_name',
            values=pivot_columns,
            aggfunc='first'
        )
        pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
        return pivot_df

    def find_top_10_strategies_and_sum_pnl(self, pivot_df, pred_column_prefix):
        """Find the top 10 strategies based on the specified y_pred_* or y_actual and sum their corresponding net_pnl_change."""
        top_10_column_name = f'top_10_net_pnl_sum_{pred_column_prefix}'
        pivot_df[top_10_column_name] = 0.0
        for index, row in pivot_df.iterrows():
            y_pred_cols = [col for col in row.index if col.startswith(f'{pred_column_prefix}_')]
            top_10_strategies = row[y_pred_cols].nlargest(10).index
            top_10_net_pnl_cols = [col.replace(pred_column_prefix, 'net_pnl_change') for col in top_10_strategies]
            pivot_df.at[index, top_10_column_name] = float(row[top_10_net_pnl_cols].sum() / 10)
        return pivot_df

    def find_bottom_10_strategies_and_sum_pnl(self, pivot_df, pred_column_prefix):
        """Find the bottom 10 strategies based on the specified y_pred_* or y_actual and sum their corresponding net_pnl_change."""
        bottom_10_column_name = f'bottom_10_net_pnl_sum_{pred_column_prefix}'
        pivot_df[bottom_10_column_name] = 0.0
        for index, row in pivot_df.iterrows():
            y_pred_cols = [col for col in row.index if col.startswith(f'{pred_column_prefix}_')]
            bottom_10_strategies = row[y_pred_cols].nsmallest(10).index
            bottom_10_net_pnl_cols = [col.replace(pred_column_prefix, 'net_pnl_change') for col in bottom_10_strategies]
            pivot_df.at[index, bottom_10_column_name] = float(row[bottom_10_net_pnl_cols].sum() / 10)
        return pivot_df

    def create_cumulative_column(self, pivot_df, pred_column_prefix, top_or_bottom):
        """Creates a cumulative sum column over time for the 'top_10_net_pnl_sum' or 'bottom_10_net_pnl_sum'."""
        pivot_df = pivot_df.sort_index()
        cumulative_column_name = f'cumulative_{top_or_bottom}_10_net_pnl_sum_{pred_column_prefix}'
        pivot_df[cumulative_column_name] = pivot_df[f'{top_or_bottom}_10_net_pnl_sum_{pred_column_prefix}'].cumsum()
        return pivot_df

    def plot_cumulative_pnl(self, final_df, periods):
        """Plots the cumulative net PnL for y_actual, y_pred_MLP, y_pred_SGD, y_pred_ABR on the same graph, along with top/bottom 10 shaded region."""
        with PdfPages(os.path.join(self.base_dir, 'combined_cumulative_net_pnl_plot_all.pdf')) as pdf:
            plt.figure(figsize=(12, 6))

            # Plot each cumulative net PnL line for y_actual, y_pred_MLP, y_pred_SGD, y_pred_ABR
            plt.plot(final_df.index, final_df['cumulative_top_10_net_pnl_sum_y_actual'], label='Actual Top 10', color='b')
            plt.plot(final_df.index, final_df['cumulative_top_10_net_pnl_sum_y_pred_MLP'], label='MLP Top 10', color='g')
            plt.plot(final_df.index, final_df['cumulative_top_10_net_pnl_sum_y_pred_SGD'], label='SGD Top 10', color='r')
            plt.plot(final_df.index, final_df['cumulative_top_10_net_pnl_sum_y_pred_ABR'], label='ABR Top 10', color='c')

            # Plot Actual Bottom 10 cumulative PnL
            plt.plot(final_df.index, final_df['cumulative_bottom_10_net_pnl_sum_y_actual'], label='Actual Bottom 10', color='darkred')

            # Shade the region between the Actual Top 10 and Actual Bottom 10 lines
            plt.fill_between(final_df.index, final_df['cumulative_bottom_10_net_pnl_sum_y_actual'], final_df['cumulative_top_10_net_pnl_sum_y_actual'], color='gray', alpha=0.3, label='Top-Bottom Spread')

            plt.title('Net PnL of Top 10 and Bottom 10 Strategies Over Time')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Net PnL')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    def run(self):
        """Executes the process for y_actual, y_pred_MLP, y_pred_SGD, and y_pred_ABR."""
        self.load_allocation_data()
        pivot_df = self.pivot_by_date()
        for pred_column_prefix in ['y_actual', 'y_pred_MLP', 'y_pred_SGD', 'y_pred_ABR']:
            pivot_df = self.find_top_10_strategies_and_sum_pnl(pivot_df, pred_column_prefix)
            pivot_df = self.create_cumulative_column(pivot_df, pred_column_prefix, 'top')
            pivot_df = self.find_bottom_10_strategies_and_sum_pnl(pivot_df, pred_column_prefix)
            pivot_df = self.create_cumulative_column(pivot_df, pred_column_prefix, 'bottom')
        self.plot_cumulative_pnl(pivot_df, pivot_df.index)
        output_path = os.path.join(self.base_dir, 'validation.csv')
        pivot_df.to_csv(output_path)
        return pivot_df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Run the process for cumulative PnL
    pivot = AllocationPivot(base_dir)
    final_df = pivot.run()

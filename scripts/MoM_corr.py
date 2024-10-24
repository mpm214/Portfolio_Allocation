import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

class MonthOverMonth:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.df = None
        self.merged_df = None

    def load_data(self):
        file_path = os.path.join(self.base_dir, 'combined_hourly_pnl_data_updated.csv')
        self.df = pd.read_csv(file_path)
        self.df['Date'] = pd.to_datetime(self.df['time'])

    def calculate_net_pnl_change(self):
        return self.df.groupby(['strategy_name', self.df['Date'].dt.to_period('M')])['net_pnl_change'].sum().reset_index().rename(columns={'Date': 'Period'})

    def calculate_ranks(self):
        result_df = self.calculate_net_pnl_change()
        all_ranks_df = pd.DataFrame()
        for period in result_df['Period'].unique():
            period_data = result_df[result_df['Period'] == period].copy()
            period_data.loc[:, 'Rank'] = period_data['net_pnl_change'].rank(ascending=False, method='first').astype(int)
            all_ranks_df = pd.concat([all_ranks_df, period_data])
        self.all_strategy_ranks = all_ranks_df

    def add_next_month_data(self):
        current_month_df = self.all_strategy_ranks.copy()
        next_month_df = self.all_strategy_ranks.copy()
        next_month_df['Period'] = next_month_df['Period'] - 1
        merged_df = pd.merge(current_month_df, next_month_df, on=['strategy_name', 'Period'], suffixes=('_current', '_next'), how='left')
        merged_df['Top_10_current'] = np.where(merged_df['Rank_current'].between(1, 10), 1, 0)
        merged_df['Top_10_next'] = np.where(merged_df['Rank_next'].between(1, 10), 1, 0)
        self.merged_df = merged_df

    def save_results(self):
        output_path = os.path.join(self.base_dir, "all_strategy_ranks.csv")
        self.merged_df.to_csv(output_path, index=False)

    def plot_and_save(self):
        cols_for_corr = ['net_pnl_change_current', 'net_pnl_change_next', 'Rank_current', 'Rank_next', 'Top_10_current', 'Top_10_next']
        corr_df = self.merged_df[cols_for_corr]
        corr_values = {
            'Net PnL Change (Current & Next)': corr_df['net_pnl_change_current'].corr(corr_df['net_pnl_change_next']),
            'Rank (Current & Next)': corr_df['Rank_current'].corr(corr_df['Rank_next']),
            'Top 10 (Current & Next)': corr_df['Top_10_current'].corr(corr_df['Top_10_next'])
        }

        pdf_path = os.path.join(self.base_dir, 'analysis_plots.pdf')
        with PdfPages(pdf_path) as pdf:

            fig, ax = plt.subplots(figsize=(6, 4))
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
            norm = plt.Normalize(-1, 1)
            # correlation coefficients
            for i, (label, value) in enumerate(corr_values.items()):
                color = cmap(norm(value))
                ax.add_patch(plt.Rectangle((0.1, 0.7 - i * 0.3), 0.8, 0.2, color=color))
                ax.text(0.5, 0.8 - i * 0.3, f'{label}: {value:.2f}', ha='center', va='center', fontsize=12, color='white')
            ax.set_axis_off()
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', fraction=0.02, pad=0.05)
            plt.title('Correlation Coefficients')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Scatter plot
            plt.figure(figsize=(8, 8))
            sns.jointplot(x='net_pnl_change_current', y='net_pnl_change_next', data=self.merged_df, kind='scatter', marginal_kws={'bins': 30})
            plt.suptitle('Scatter Plot')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    def run_analysis(self):
        self.load_data()
        self.calculate_ranks()
        self.add_next_month_data()
        self.save_results()
        self.plot_and_save()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    analyzer = MonthOverMonth(base_dir)
    analyzer.run_analysis()

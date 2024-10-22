import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class PortfolioAnalyzer:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.pnl_data = pd.read_csv(self.input_path)
        self.pnl_data['Date'] = pd.to_datetime(self.pnl_data['time'])
        self.pnl_data_processed = None

    def process_strategy(self):
        def _process(group):
            group = group.sort_values(by='Date').reset_index(drop=True)
            group['normalized_net_pnl'] = group['net_pnl'] - group['net_pnl'].iloc[0]
            group['daily_return'] = group['normalized_net_pnl'].pct_change()
            return group.dropna(subset=['daily_return'])
        
        self.pnl_data_processed = self.pnl_data.groupby('strategy_name').apply(_process).reset_index(drop=True)

    def plot_strategies(self):
        strategies = self.pnl_data_processed['strategy_name'].unique()
        num_strategies = len(strategies)
        batch_size = 10
        pdf_path = os.path.join(self.output_path, 'fixed_strategy_plots.pdf')
        
        with PdfPages(pdf_path) as pdf:
            for i in range(0, num_strategies, batch_size):
                plt.figure(figsize=(14, 8))
                for strategy in strategies[i:i + batch_size]:
                    strategy_data = self.pnl_data_processed[self.pnl_data_processed['strategy_name'] == strategy]
                    plt.plot(strategy_data['Date'], strategy_data['normalized_net_pnl'], label=strategy)
                
                plt.xlabel('Date')
                plt.ylabel('Normalized Net PnL')
                plt.title(f'Normalized Net PnL Over Date for Strategies {i + 1} to {i + min(batch_size, num_strategies - i)}')
                plt.legend(loc='best')
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                pdf.savefig()
                plt.close()

    def run_analysis(self):
        self.process_strategy()
        self.plot_strategies()

if __name__ == "__main__":
    input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined_hourly_pnl_data_updated.csv')
    output_path = os.path.dirname(os.path.abspath(__file__))
    analyzer = PortfolioAnalyzer(input_path, output_path)
    analyzer.run_analysis()

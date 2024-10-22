import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class PortfolioAnalyzer:
    def __init__(self, input_paths, strategies):
        self.input_paths = input_paths
        self.strategies = strategies
        self.pnl_data_before = pd.read_csv(self.input_paths['before'])
        self.pnl_data_after = pd.read_csv(self.input_paths['after'])
        self.pnl_data_before['Date'] = pd.to_datetime(self.pnl_data_before['time'])
        self.pnl_data_after['Date'] = pd.to_datetime(self.pnl_data_after['time'])

    def filter_strategies(self, data):
        return data[data['strategy_name'].isin(self.strategies)]

    def process_strategy(self, data):
        def _process(group):
            group = group.sort_values(by='Date').reset_index(drop=True)
            group['normalized_net_pnl'] = group['net_pnl'] - group['net_pnl'].iloc[0]
            return group
        
        return data.groupby('strategy_name').apply(_process).reset_index(drop=True)

    def plot_strategies(self):
        filtered_data_before = self.filter_strategies(self.pnl_data_before)
        filtered_data_after = self.filter_strategies(self.pnl_data_after)

        pnl_data_processed_before = self.process_strategy(filtered_data_before)
        pnl_data_processed_after = self.process_strategy(filtered_data_after)

        pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Appendix_Anomaly.pdf')
        with PdfPages(pdf_path) as pdf:
            plt.figure(figsize=(14, 16))

            # Plot before anomaly fix
            ax1 = plt.subplot(2, 1, 1)
            for strategy in self.strategies:
                strategy_data = pnl_data_processed_before[pnl_data_processed_before['strategy_name'] == strategy]
                ax1.plot(strategy_data['Date'], strategy_data['normalized_net_pnl'], label=strategy)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Normalized Net PnL')
            ax1.set_title('Normalized Net PnL Before Anomaly Fix')
            ax1.legend(loc='best', fontsize='small')
            ax1.grid(True)
            ax1.tick_params(axis='x', rotation=45)

            # Plot after anomaly fix
            ax2 = plt.subplot(2, 1, 2)
            for strategy in self.strategies:
                strategy_data = pnl_data_processed_after[pnl_data_processed_after['strategy_name'] == strategy]
                ax2.plot(strategy_data['Date'], strategy_data['normalized_net_pnl'], label=strategy)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Normalized Net PnL')
            ax2.set_title('Normalized Net PnL After Anomaly Fix')
            ax2.legend(loc='best', fontsize='small')
            ax2.grid(True)
            ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

    def run_analysis(self):
        self.plot_strategies()

if __name__ == "__main__":
    input_paths = {
        'before': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined_pnl_data.csv'),
        'after': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined_hourly_pnl_data_updated.csv')
    }
    strategies = [
        'L_GBPUSD_S34_2_1227018_49061A95',
        'L_GBPUSD_S34_4_8079186_743600CC',
        'L_GBPUSD_S42_0_1097601_6C61C210',
        'L_GBPUSD_S34_3_6845667_DC6B39EE',
        'L_GBPUSD_S34_2_9722597_DB736AAF',
        'L_GBPUSD_S42_2_1263481_012B54F7',
        'L_GBPUSD_S42_4_1672550_C717C9EF',
        'L_GBPUSD_S42_4_1061317_58ADC376',
        'L_GBPUSD_S42_4_1248147_8DDEB5ED',
        'L_GBPUSD_S42_4_793519_D8FAB916',
        'S_GBPUSD_S35_1_3749132_746299E1',
        'S_GBPUSD_S35_1_6667748_CD1C245A',
        'S_GBPUSD_S35_1_7791031_A9EA0170',
        'S_GBPUSD_S35_1_5478144_D156D8AD',
        'S_GBPUSD_S35_2_5049397_D5A19A22',
        'S_GBPUSD_S35_0_2648124_940EFDBB',
        'S_GBPUSD_S35_3_6866072_43587EAA',
        'S_GBPUSD_S35_3_4974188_B011F908',
        'S_GBPUSD_S43_0_385891_C10014B8',
        'S_GBPUSD_S43_2_2540157_E8ED90A3',
        'S_GBPUSD_S43_3_2447853_CDFA30F9',
        'S_GBPUSD_S43_4_1314974_DA658E5F',
        'S_GBPUSD_S43_4_2441780_37DA97F5',
        'S_GBPUSD_S43_4_49961_9510B871'
    ]
    analyzer = PortfolioAnalyzer(input_paths, strategies)
    analyzer.run_analysis()
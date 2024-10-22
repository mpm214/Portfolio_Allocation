import os
import pandas as pd
from scipy.stats import kstest
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class XVariableTester:
    def __init__(self, filename):
        # Get the script directory dynamically and set the full path
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.join(self.script_dir, filename)
        self.data = None
        self.load_data()
    
    def load_data(self):
        # Load the CSV data
        try:
            self.data = pd.read_csv(self.filepath)
            print(f"Data loaded successfully from {self.filepath}")

            # For UNDERLYING METRICS ONLY
            columns_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume']
            for column in columns_to_drop:
                if column in self.data.columns:
                    self.data.drop(columns=[column], inplace=True)
                    print(f"Column '{column}' removed successfully.")

            '''# FOR STRATEGY_METRICS ONLY: Drop the 'net_pnl' column if it exists
            if 'net_pnl' in self.data.columns:
                self.data.drop(columns=['net_pnl'], inplace=True)
                print("Column 'net_pnl' removed successfully.")'''
        
        except FileNotFoundError:
            print(f"File {self.filepath} not found.")
    
    def handle_missing_values(self):
        # Select only numeric columns for checking NaN or infinite values
        numeric_cols = self.data.select_dtypes(include=[float, int])

        # Check for NaN or infinite values in the numeric columns only
        if numeric_cols.isnull().values.any() or np.isinf(numeric_cols).values.any():
            print("Missing or infinite values found, handling them by dropping rows...")
            # Drop rows with NaN or infinite values in numeric columns only
            self.data = self.data.replace([np.inf, -np.inf], np.nan).dropna()

    def calculate_vif(self):
        # Handle missing values before calculating VIF
        self.handle_missing_values()
        
        X = self.data.select_dtypes(include=[float, int])  # Ensure only numerical columns are included
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data

    def test_normality(self):
        # Perform Kolmogorov-Smirnov Test (Normality test)
        normality_results = {}

        for column in self.data.select_dtypes(include=[float, int]).columns:
            # Standardize the data before testing
            standardized_data = (self.data[column] - self.data[column].mean()) / self.data[column].std()
            ks_stat, p_value = kstest(standardized_data, 'norm')  # Test against the normal distribution

            # Store the results
            normality_results[column] = p_value

        # Convert the results into a DataFrame for easy analysis
        normality_df = pd.DataFrame(list(normality_results.items()), columns=['Variable', 'KS_p_value'])
        return normality_df

    
    def test_stationarity(self):
        # Perform Augmented Dickey-Fuller Test (Stationarity test)
        stationarity_results = []

        for column in self.data.select_dtypes(include=[float, int]).columns:
            adf_result = adfuller(self.data[column], autolag='AIC')
            p_value = adf_result[1]  # Extract p-value
            lag_used = adf_result[2]  # Extract the number of lags used

            # Append the results to the list
            stationarity_results.append({
                'Variable': column,
                'ADF_p_value': p_value,
                'Lags_Used': lag_used
            })

        # Convert the results into a DataFrame for easy analysis
        stationarity_df = pd.DataFrame(stationarity_results)
        return stationarity_df


    
    def plot_autocorrelation(self, pdf):
        # Plot autocorrelation for each numerical variable and save to PDF
        for column in self.data.select_dtypes(include=[float, int]).columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_acf(self.data[column], lags=20, ax=ax)
            ax.set_title(f"Autocorrelation for {column}")
            ax.set_xlabel("Lags")
            ax.set_ylabel("Autocorrelation")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    
    def run_all_tests(self):
        # Create output file paths in the script's directory
        pdf_path = os.path.join(self.script_dir, 'var_testing_plots.pdf')
        csv_path = os.path.join(self.script_dir, 'var_testing_summary.csv')

        # Create a PdfPages object to save plots
        with PdfPages(pdf_path) as pdf:
            # Plot autocorrelation graphs and export them to the PDF
            self.plot_autocorrelation(pdf)
        
        # Run VIF calculation and save to CSV
        vif_df = self.calculate_vif()
        vif_df.to_csv(csv_path, index=False)
        
        # Run Normality Test and save to CSV (append mode)
        normality_df = self.test_normality()
        normality_df.to_csv(csv_path, mode='a', index=False)
        
        # Run Stationarity Test and save to CSV (append mode)
        stationarity_df = self.test_stationarity()
        stationarity_df.to_csv(csv_path, mode='a', index=False)

# Example usage
if __name__ == "__main__":
    tester = XVariableTester('GBPUSD_Underlying_Hour_metrics.csv') #'strategy_ratios.csv' #'strategy_metrics.csv' #'EURUSD_Underlying_Hour_metrics.csv'
    tester.run_all_tests()

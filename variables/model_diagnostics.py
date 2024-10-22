import pandas as pd
import statsmodels.api as sm
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
import csv
import io

class OLSCollinearityDiagnostics:
    def __init__(self, file_name, y_column, columns_to_remove=None, output_csv='ols_diagnostics_output.csv'):
        """
        Initializes the class for OLS regression and collinearity diagnostics.
        :param file_name: Path to the CSV file containing the data
        :param y_column: The target column for OLS regression
        :param columns_to_remove: List of columns to remove before processing
        :param output_csv: Output CSV file to store the results
        """
        self.file_name = file_name
        self.y_column = y_column
        self.columns_to_remove = columns_to_remove or ['net_pnl', 'net_pnl_change', 'net_pnl_change_y', 'net_pnl_change_x', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.output_csv = output_csv
        self.df = None
        self.X = None
        self.y = None
        self.model = None

    def load_and_clean_data(self):
        """
        Loads the data from CSV and cleans it by removing unnecessary columns and handling missing values.
        """
        self.df = pd.read_csv(self.file_name)
        self.df.drop(columns=self.columns_to_remove, inplace=True, errors='ignore')
        self.df = self.df.select_dtypes(include=['number'])  # Keep only numeric columns
        self.df.dropna(inplace=True)

        # Set X and y
        self.y = self.df[self.y_column]
        self.X = self.df.drop(columns=[self.y_column])

    def fit_ols(self):
        """
        Fits an OLS regression model and writes the summary to the CSV file.
        """
        self.X = sm.add_constant(self.X)  # Add constant for intercept
        self.model = sm.OLS(self.y, self.X).fit()

        # Capture the OLS summary into a string buffer
        summary = self.model.summary()
        with open(self.output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["OLS Regression Results"])
            writer.writerow([summary.as_text()])

    def calculate_vif(self):
        """
        Calculates Variance Inflation Factor (VIF) to check for multicollinearity and writes it to the CSV file.
        """
        vif_data = pd.DataFrame()
        vif_data["feature"] = self.X.columns
        vif_data["VIF"] = [variance_inflation_factor(self.X.values, i) for i in range(self.X.shape[1])]

        # Append VIF data to the CSV file
        with open(self.output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([])
            writer.writerow(["Variance Inflation Factors (VIF)"])
            writer.writerow(["Feature", "VIF"])
            for i in range(len(vif_data)):
                writer.writerow([vif_data['feature'][i], vif_data['VIF'][i]])

    def run_diagnostics(self):
        """
        Complete diagnostic process: loading data, fitting OLS, and calculating VIF for multicollinearity.
        """
        self.load_and_clean_data()
        self.fit_ols()
        self.calculate_vif()

# Example usage:
file_name = os.path.join(os.path.dirname(__file__), 'full.csv')
y_column = 'Rank'
columns_to_remove = ['net_pnl', 'net_pnl_change', 'net_pnl_change_y', 'net_pnl_change_x', 'Open', 'High', 'Low', 'Close', 'Volume']
output_csv = os.path.join(os.path.dirname(__file__), 'ols_diagnostics_output.csv')

# Create an instance of OLSCollinearityDiagnostics and run diagnostics
ols_diagnostics = OLSCollinearityDiagnostics(file_name, y_column, columns_to_remove, output_csv)
ols_diagnostics.run_diagnostics()

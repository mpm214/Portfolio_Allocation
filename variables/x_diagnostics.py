import pandas as pd
import statsmodels.api as sm
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
import random
from tqdm import tqdm  # Import tqdm for progress bar

class OLSCollinearityDiagnostics:
    def __init__(self, file_name, y_column, strategy_column='strategy_name', columns_to_remove=None, output_excel='ols_diagnostics_output.xlsx', random_seed=42):
        """
        Initializes the class for OLS regression and collinearity diagnostics.
        :param file_name: Path to the CSV file containing the data
        :param y_column: The target column for OLS regression
        :param strategy_column: The column that contains the strategy names to filter by
        :param columns_to_remove: List of columns to remove before processing
        :param output_excel: Output Excel file to store the results
        :param random_seed: Seed for the random number generator to ensure reproducibility
        """
        self.file_name = file_name
        self.y_column = y_column
        self.strategy_column = strategy_column
        self.columns_to_remove = columns_to_remove or ['Long_Short', 'net_pnl', 'net_pnl_change', 'net_pnl_change_y', 'net_pnl_change_x', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.output_excel = output_excel
        self.random_seed = random_seed  # Set the random seed
        self.df = None
        self.X = None
        self.y = None
        self.model = None
        self.writer = pd.ExcelWriter(self.output_excel, engine='xlsxwriter')  # Proper import of ExcelWriter
        self.summary_data = []  # To hold VIF results for summary page
        self.strategy_info = []  # List to store strategy name and stats for the "Strategy_Info" sheet

    def load_data(self):
        """Loads the data from CSV."""
        self.df = pd.read_csv(self.file_name, low_memory=False)  # Added low_memory=False to avoid dtype warnings
        self.df.drop(columns=self.columns_to_remove, inplace=True, errors='ignore')
        self.df = self.df.select_dtypes(include=['number', 'object'])  # Keep numeric and object columns (for strategy filtering)
        self.df.dropna(inplace=True)

    def filter_by_strategy(self, strategy):
        """Filter the dataset for a given strategy."""
        filtered_df = self.df[self.df[self.strategy_column] == strategy]

        # Drop non-numeric columns after filtering
        filtered_df = filtered_df.select_dtypes(include=['number'])

        self.y = filtered_df[self.y_column]
        self.X = filtered_df.drop(columns=[self.y_column])

    def fit_ols(self, sheet_name):
        """Fits an OLS regression model and saves the summary to a sheet."""
        self.X = sm.add_constant(self.X)  # Add constant for intercept
        self.model = sm.OLS(self.y, self.X).fit()

        # Save OLS summary to the Excel sheet
        summary_text = self.model.summary().as_text()

        # Write summary text to the sheet by converting it into DataFrame format
        summary_lines = summary_text.splitlines()
        summary_df = pd.DataFrame([line.split() for line in summary_lines if line])

        summary_df.to_excel(self.writer, sheet_name=str(sheet_name), index=False, header=False)

    def calculate_vif(self, sheet_name):
        """Calculates Variance Inflation Factor (VIF) and writes to the Excel file."""
        vif_data = pd.DataFrame()
        vif_data["feature"] = self.X.columns
        vif_data["VIF"] = [variance_inflation_factor(self.X.values, i) for i in range(self.X.shape[1])]

        # Append VIF results to summary data
        self.summary_data.append(vif_data)

        # Write VIF data to the Excel sheet
        vif_data.to_excel(self.writer, sheet_name=str(sheet_name), startrow=80, index=False)

    def run_diagnostics(self, strategy, sheet_name):
        """Complete diagnostic process for a single strategy."""
        self.filter_by_strategy(strategy)
        
        # Collect stats for the y-column (Rank)
        strategy_stats = {
            "Sheet": sheet_name,
            "Strategy": strategy,
            "Max Rank": self.y.max(),
            "Min Rank": self.y.min(),
            "Avg Rank": self.y.mean()
        }
        self.strategy_info.append(strategy_stats)  # Add to strategy info list

        self.fit_ols(sheet_name=sheet_name)
        self.calculate_vif(sheet_name=sheet_name)

    def run_multiple_diagnostics(self, n=5):
        """Run diagnostics for `n` randomly selected strategies and save to different sheets."""
        self.load_data()

        strategies = self.df[self.strategy_column].unique()

        # Set the random seed to ensure reproducibility
        random.seed(self.random_seed)

        random_strategies = random.sample(list(strategies), n)

        # Add tqdm progress bar here
        for i, strategy in enumerate(tqdm(random_strategies, desc="Running Diagnostics")):
            sheet_name = i + 1  # Simply use numeric sheet names: 1, 2, 3, etc.
            self.run_diagnostics(strategy, sheet_name=sheet_name)

        self.calculate_summary()

    def calculate_summary(self):
        """Calculate and write summary of key metrics to the Excel file."""
        vif_concat = pd.concat(self.summary_data)
        vif_mean = vif_concat.groupby('feature').mean().reset_index()

        # Write the summary to a new sheet
        vif_mean.to_excel(self.writer, sheet_name='Summary', index=False)

        # Create a DataFrame for strategy information and write it to a new sheet
        strategy_info_df = pd.DataFrame(self.strategy_info)
        strategy_info_df.to_excel(self.writer, sheet_name='Strategy_Info', index=False)

    def save(self):
        """Saves the Excel file with all the sheets."""
        self.writer.close()  # Use close instead of save

# Example usage:
file_name = os.path.join(os.path.dirname(__file__), 'full.csv')
y_column = 'Rank'
strategy_column = 'strategy_name'  # Column containing strategy names
columns_to_remove = ['Long_Short', 'net_pnl', 'net_pnl_change', 'net_pnl_change_y', 'net_pnl_change_x', 'Open', 'High', 'Low', 'Close', 'Volume']
output_excel = os.path.join(os.path.dirname(__file__), 'ols_diagnostics_output.xlsx')

# Create an instance of OLSCollinearityDiagnostics and run diagnostics 5 times
ols_diagnostics = OLSCollinearityDiagnostics(file_name, y_column, strategy_column, columns_to_remove, output_excel, random_seed=42)
ols_diagnostics.run_multiple_diagnostics(n=10)
ols_diagnostics.save()

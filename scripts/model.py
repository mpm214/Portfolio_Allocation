import pandas as pd
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import AdaBoostRegressor
from tqdm import tqdm
from walk_through_validation import PeriodGenerator

class PortfolioModeling:
    def __init__(self, max_strategies=None):
        """
        Initialize the model with the option to limit the number of strategies to run.
        :param max_strategies: The number of strategies to run. If None, run all strategies.
        """
        # Dynamically retrieve the path where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.full_file = os.path.join(script_dir, 'full.csv')
        self.max_strategies = max_strategies
        self.load_data()

    def load_data(self):
        """Loads and preprocesses the full dataset with specified dtypes and selected columns."""
        columns_to_keep = [
            'Date', 'strategy_name', 'Rank', 'Long_Short', 'Cumulative_VWAP', 'Cumulative_Volume', 'Precision',
            'Combined_Ratio', 'Downside_Ratio', 'Lower_Band', 'Band_Width', 'Upper_Band', 'Rolling_Mean', 
            'rolling_Peak_30', 'Close', 'EMA_4800', 'EMA_1200', 'Crossover', 'SMA_4800', 'CrossUp', 'PSAR', 
            'EMA_240', 'MACD', 'PSAR_direction', 'L14', 'PnL_slope_30', 'H14', 'TR_Smooth', 'SMA_1200', 
            'DM_down_Smooth', 'DM_up_Smooth', 'DI_up', 'Rolling_Max_Recovery_Time', 'DI_down', 'rolling_std_30', 
            'sharpe_ratio', 'OBV', 'K_PCT', 'Drawdown', 'CMF'
        ]

        # Define data types
        dtype_dict = {
            'Date': 'object',
            'strategy_name': 'object',
            'Long_Short': 'int64',
            'Rank': 'float64',
            'Cumulative_VWAP': 'float64', 'Cumulative_Volume': 'float64', 'Precision': 'float64',
            'Combined_Ratio': 'float64', 'Downside_Ratio': 'float64', 'Lower_Band': 'float64',
            'Band_Width': 'float64', 'Upper_Band': 'float64', 'Rolling_Mean': 'float64',
            'rolling_Peak_30': 'float64', 'Close': 'float64', 'EMA_4800': 'float64', 'EMA_1200': 'float64',
            'Crossover': 'float64', 'SMA_4800': 'float64', 'CrossUp': 'float64', 'PSAR': 'float64',
            'EMA_240': 'float64', 'MACD': 'float64', 'PSAR_direction': 'float64', 'L14': 'float64',
            'PnL_slope_30': 'float64', 'H14': 'float64', 'TR_Smooth': 'float64', 'SMA_1200': 'float64',
            'DM_down_Smooth': 'float64', 'DM_up_Smooth': 'float64', 'DI_up': 'float64',
            'Rolling_Max_Recovery_Time': 'float64', 'DI_down': 'float64', 'rolling_std_30': 'float64',
            'sharpe_ratio': 'float64', 'OBV': 'float64', 'K_PCT': 'float64', 'Drawdown': 'float64',
            'CMF': 'float64'
        }

        # Load data with specified dtypes and only necessary columns
        self.data = pd.read_csv(self.full_file, usecols=columns_to_keep, dtype=dtype_dict, low_memory=False)

        # Ensure 'Date' column is in datetime format with hours
        self.data['Date'] = pd.to_datetime(self.data['Date'], utc=True)
        self.data.sort_values(by=['Date', 'strategy_name'], inplace=True)

    def walk_forward_validation(self, x_data, y_data, periods, strategy_name):
        """Executes walk-forward validation using the provided pipeline."""
        # Ensure 'Date' column is in datetime format
        x_data.loc[:, 'Date'] = pd.to_datetime(x_data['Date'], utc=True)
        y_data.loc[:, 'Date'] = pd.to_datetime(y_data['Date'], utc=True)

        x_slices = periods.slice_data_by_period(x_data, 'Date')
        y_slices = periods.slice_data_by_period(y_data, 'Date')

        models = [
            ('MLP', MLPRegressor(max_iter=1000, random_state=42)),
            ('SGD', SGDRegressor(random_state=42)),
            ('ABR', AdaBoostRegressor(random_state=42))
        ]

        predictions = []

        for (x_train, x_test), (y_train, y_test) in tqdm(zip(x_slices, y_slices), total=len(x_slices), desc=f'Processing periods for {strategy_name}'):
            X_train = x_train.drop(columns=['Date', 'strategy_name'])
            Y_train = y_train['Rank']
            X_test = x_test.drop(columns=['Date', 'strategy_name'])
            Y_test = y_test['Rank']

            # Standardize the features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Create a dictionary to store the model-specific columns
            period_data = {'Date': x_test['Date'].values, 'strategy_name': strategy_name}
            
            # Loop through each model and add its predictions to the period_data dictionary
            for name, model in models:
                model.fit(X_train, Y_train)
                y_pred = model.predict(X_test)

                # Add y_actual and y_pred for this model as new columns
                period_data[f'y_actual_{name}'] = Y_test.values
                period_data[f'y_pred_{name}'] = y_pred

            # Create DataFrame for this period's data, including all models
            period_predictions = pd.DataFrame(period_data)
            predictions.append(period_predictions)

        return pd.concat(predictions)

    def shift_and_pivot_results(self, final_predictions):
        """Modifies the Date to Year-Month format, shifts it forward, and pivots the data."""
        # Step 1: Modify Date to Year-Month format
        final_predictions['Date'] = final_predictions['Date'].dt.to_period('M').dt.to_timestamp()

        # Step 2: Shift the Date forward by one month
        final_predictions['Date'] = final_predictions['Date'] + pd.DateOffset(months=1)

        # Step 3: Group by 'Date' and 'strategy_name' and calculate the mean for all 'y_actual_*' and 'y_pred_*' columns
        y_columns = [col for col in final_predictions.columns if col.startswith('y_actual') or col.startswith('y_pred')]
        
        pivot_table = final_predictions.groupby(['Date', 'strategy_name'])[y_columns].mean().reset_index()

        return pivot_table

    def execute(self):
        """Runs the entire workflow for all strategies or a limited number of strategies."""
        all_predictions = []
        unique_strategies = self.data['strategy_name'].unique()

        # If max_strategies is specified, limit the number of strategies to that amount
        if self.max_strategies is not None:
            unique_strategies = unique_strategies[:self.max_strategies]

        # Add a tqdm progress bar for processing strategies
        for strategy in tqdm(unique_strategies, desc='Processing strategies'):
            strategy_data = self.data[self.data['strategy_name'] == strategy].reset_index(drop=True)

            x = strategy_data
            y = strategy_data[['Date', 'Rank']]

            # Initialize PeriodGenerator from the external file with hours included
            start_date = '2021-06-01 00:00'
            end_date = '2023-12-31 23:00'
            period_generator = PeriodGenerator(start_date, end_date)

            strategy_predictions = self.walk_forward_validation(x, y, period_generator, strategy)
            all_predictions.append(strategy_predictions)

        # Combine all predictions into a single DataFrame
        final_predictions = pd.concat(all_predictions)

        # Step 4: Modify Date, shift it forward by one month, and pivot the table
        final_predictions_pivoted = self.shift_and_pivot_results(final_predictions)

        # Save the pivoted predictions to CSV
        output_file = os.path.join(os.path.dirname(self.full_file), 'results.csv')
        final_predictions_pivoted.to_csv(output_file, index=False)



if __name__ == "__main__":
    portfolio_modeling = PortfolioModeling(max_strategies=None)  # Set to None to run all strategies
    portfolio_modeling.execute()

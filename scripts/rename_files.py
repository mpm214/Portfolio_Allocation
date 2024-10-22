import os
import pandas as pd

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(('-pnl.csv', '-positions.csv')):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            strategy_name = df['strategy_name'].iloc[0]
            new_filename = f"{strategy_name}-{filename.split('-')[-1]}"
            os.rename(file_path, os.path.join(directory, new_filename))
            print(f"Renamed '{filename}' to '{new_filename}'")

directory_path = r'Data\forex-dynamic-portfolio-allocation' # update path
rename_files(directory_path)
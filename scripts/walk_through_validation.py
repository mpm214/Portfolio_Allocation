import os
import pandas as pd
import numpy as np
from datetime import timedelta

class PeriodGenerator:
    def __init__(self, start_date, end_date):
        self.start_date = pd.to_datetime(start_date, utc=True)
        self.end_date = pd.to_datetime(end_date, utc=True) 

    def generate_periods(self):
        """Generate periods for walk-forward validation with 1-year training and 1-month test."""
        periods = []
        current_start = self.start_date
        final_end = self.end_date
        
        while current_start <= final_end:
            # Ensure train_start is the start of the day (00:00)
            train_start = current_start
            train_end = (train_start + pd.DateOffset(years=1) - timedelta(hours=1))
            test_start = (train_end + timedelta(hours=1)).replace(hour=0, minute=0)
            test_end = (test_start + pd.DateOffset(months=1) - timedelta(hours=1))
            
            if test_end > final_end:
                break

            periods.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            # Move the current start by one month for the next period
            current_start += pd.DateOffset(months=1)
        
        return periods

    def slice_data_by_period(self, data, date_col):
        """Slice data by the generated periods."""
        periods = self.generate_periods()
        slices = []
        
        for period in periods:
            # Filter the data for each period based on the date column
            train_data = data[(data[date_col] >= period['train_start']) & (data[date_col] <= period['train_end'])]
            test_data = data[(data[date_col] >= period['test_start']) & (data[date_col] <= period['test_end'])]
            slices.append((train_data, test_data))
        
        return slices

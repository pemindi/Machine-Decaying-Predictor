import pandas as pd
import numpy as np
from datetime import datetime

class DataProcessor:
    def __init__(self):
        # Always use current date - no override needed
        self.today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    def datetime_to_elapsed_days(self, dt_series):
        """Convert datetime series to elapsed days from start."""
        start = dt_series.iloc[0]
        return (dt_series - start).dt.total_seconds() / (3600 * 24)

    def prepare_data(self, df, rms_threshold=0.01, apply_minimal_filtering=False):
        """Prepare and filter data."""
        print(f"Input data shape: {df.shape}")
        
        # Convert datetime column
        try:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
        except:
            try:
                df['Datetime'] = pd.to_datetime(df['Datetime'], format='%m/%d/%Y %H:%M')
            except:
                df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')

        # Convert RUL to numeric if needed
        if 'Predicted RUL' in df.columns and df['Predicted RUL'].dtype == 'object':
            df['Predicted RUL'] = pd.to_numeric(df['Predicted RUL'], errors='coerce')

        if apply_minimal_filtering:
            # Only remove rows with invalid/missing data when preprocessing is disabled
            df_filtered = df.dropna(subset=['Predicted RUL', 'Datetime']).copy()
            # Only filter extremely low RMS values (likely sensor errors)
            df_filtered = df_filtered[df_filtered['RMS'] > 0.001].copy()
        else:
            # Apply original filtering logic
            df_filtered = df[df['RMS'] > rms_threshold].copy()
            # Also remove any rows with missing critical data
            df_filtered = df_filtered.dropna(subset=['Predicted RUL', 'Datetime']).copy()

        # Sort by datetime ascending
        df_filtered = df_filtered.sort_values('Datetime').reset_index(drop=True)
        
        print(f"After filtering shape: {df_filtered.shape}")
        print(f"Datetime column length: {len(df_filtered['Datetime'])}")
        print(f"RUL column length: {len(df_filtered['Predicted RUL'])}")
        
        return df_filtered

    def create_incremental_batches(self, data, batch_size, direction='forward'):
        """Create truly incremental batches (100, 200, 300, ..., total_rows)."""
        num_rows = len(data)
        batches = []
        
        print(f"Creating batches from {num_rows} rows with batch_size {batch_size}")

        # Adjust batch size if larger than dataset
        if batch_size > num_rows:
            batch_size = num_rows

        current_size = batch_size
        while current_size <= num_rows:
            if direction == 'forward':
                batch_data = data.iloc[:current_size].copy()
                batch_info = f"Forward - First {current_size} rows"
            else:
                batch_data = data.iloc[-current_size:].copy()
                batch_info = f"Backward - Last {current_size} rows"

            # Ensure batch_data is clean
            batch_data = batch_data.dropna(subset=['Predicted RUL', 'Datetime']).reset_index(drop=True)
            
            # Date range for the batch
            start_date = batch_data['Datetime'].min().strftime('%Y-%m-%d')
            end_date = batch_data['Datetime'].max().strftime('%Y-%m-%d')
            date_range = f"({start_date} to {end_date})"
            
            print(f"Batch {current_size}: {len(batch_data)} rows after cleaning")

            batches.append({
                'data': batch_data,
                'size': len(batch_data),  # Use actual size after cleaning
                'info': f"{batch_info} {date_range}",
                'date_range': date_range
            })

            current_size += batch_size

        # Ensure final batch includes all rows if necessary
        if (num_rows % batch_size) != 0 and (batches == [] or batches[-1]['size'] != num_rows):
            if direction == 'forward':
                batch_data = data.iloc[:num_rows].copy()
                batch_info = f"Forward - First {num_rows} rows"
            else:
                batch_data = data.iloc[-num_rows:].copy()
                batch_info = f"Backward - Last {num_rows} rows"

            # Ensure batch_data is clean
            batch_data = batch_data.dropna(subset=['Predicted RUL', 'Datetime']).reset_index(drop=True)
            
            start_date = batch_data['Datetime'].min().strftime('%Y-%m-%d')
            end_date = batch_data['Datetime'].max().strftime('%Y-%m-%d')
            date_range = f"({start_date} to {end_date})"
            
            print(f"Final batch: {len(batch_data)} rows after cleaning")

            batches.append({
                'data': batch_data,
                'size': len(batch_data),  # Use actual size after cleaning
                'info': f"{batch_info} {date_range}",
                'date_range': date_range
            })

        return batches

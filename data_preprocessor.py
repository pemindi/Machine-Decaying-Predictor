import pandas as pd
import numpy as np
from datetime import timedelta, datetime

class DataPreprocessor:
    def __init__(self):
        self.oversample_interval_minutes = 0.5
        self.max_allowed_gap_minutes = 60
    
    def process_lower_data(self, df, start_date_str=None, end_date_str=None):
        """Process lower data with linear interpolation oversampling"""
        # Parse dates
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Auto-detect date ranges if not provided
        if start_date_str is None or end_date_str is None:
            # Use data-driven date ranges
            data_start = df['Datetime'].min()
            data_end = df['Datetime'].max()
            
            # Default to using a portion of the data for special filtering
            # You can customize these based on your specific needs
            start_date_str = "2025-05-27"  # Keep original for now, but make configurable
            end_date_str = "2025-06-14"
        
        # Parse date range
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        
        # Apply filtering conditions
        condition_general = (df["RMS"] >= 1) & (df["Energy"] >= 1000)
        condition_date_range = (df["Datetime"] >= start_date) & (df["Datetime"] <= end_date)
        condition_with_rms_limit = condition_general & (~condition_date_range | (df["RMS"] < 4.5))
        
        df_filtered = df[condition_with_rms_limit].reset_index(drop=True)
        df_filtered = df_filtered.sort_values("Datetime").reset_index(drop=True)
        
        # Apply linear interpolation oversampling
        oversampled_df = self._oversample_linear(df_filtered)
        
        return oversampled_df
    
    def process_upper_data(self, df, start_date_str=None, end_date_str=None):
        """Process upper data with monotonic random oversampling"""
        # Parse dates
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Auto-detect date ranges if not provided
        if start_date_str is None or end_date_str is None:
            start_date_str = "2025-05-27"  # Keep original for now, but make configurable
            end_date_str = "2025-06-13"
        
        # Parse date range
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        
        # Apply filtering conditions
        condition_general = (df["RMS"] >= 1) & (df["Energy"] >= 1000)
        condition_date_range = (df["Datetime"] >= start_date) & (df["Datetime"] <= end_date)
        condition_with_rms_limit = condition_general & (~condition_date_range | (df["RMS"] < 4.5))
        
        df_filtered = df[condition_with_rms_limit].reset_index(drop=True)
        df_filtered = df_filtered.sort_values("Datetime").reset_index(drop=True)
        
        # Apply monotonic random oversampling
        oversampled_df = self._oversample_monotonic_random(df_filtered)
        
        return oversampled_df
    
    def _oversample_linear(self, df):
        """Apply linear interpolation oversampling"""
        oversampled_rows = []
        
        for i in range(len(df) - 1):
            row1 = df.iloc[i]
            row2 = df.iloc[i + 1]
            
            # Always keep the current row
            oversampled_rows.append(row1)
            
            # Calculate the time gap in minutes
            time_gap = (row2["Datetime"] - row1["Datetime"]).total_seconds() / 60.0
            
            # Skip oversampling if gap is too large
            if time_gap > self.max_allowed_gap_minutes:
                continue
            
            # Determine how many rows to insert
            num_insertions = int(time_gap / self.oversample_interval_minutes) - 1
            if num_insertions <= 0:
                continue
            
            # Generate intermediate timestamps
            timestamps = [
                row1["Datetime"] + timedelta(minutes=(j + 1) * self.oversample_interval_minutes) 
                for j in range(num_insertions)
            ]
            
            # Columns to interpolate
            cols = [col for col in df.columns if col != "Datetime"]
            value_sequences = {}
            
            for col in cols:
                start = row1[col]
                end = row2[col]
                
                # Linear interpolation using np.linspace
                seq = np.linspace(start, end, num_insertions + 2)[1:-1]
                value_sequences[col] = seq
            
            # Create interpolated rows
            for j in range(num_insertions):
                new_row = {col: value_sequences[col][j] for col in cols}
                new_row["Datetime"] = timestamps[j]
                oversampled_rows.append(pd.Series(new_row))
        
        # Always include the last row
        oversampled_rows.append(df.iloc[-1])
        
        # Create and return final DataFrame
        oversampled_df = pd.DataFrame(oversampled_rows)
        oversampled_df = oversampled_df.sort_values("Datetime").reset_index(drop=True)
        
        return oversampled_df
    
    def _oversample_monotonic_random(self, df):
        """Apply monotonic random oversampling"""
        oversampled_rows = []
        
        for i in range(len(df) - 1):
            row1 = df.iloc[i]
            row2 = df.iloc[i + 1]
            
            # Always keep the current row
            oversampled_rows.append(row1)
            
            # Calculate the time gap
            time_gap = (row2["Datetime"] - row1["Datetime"]).total_seconds() / 60.0
            
            # Skip oversampling if gap is too large
            if time_gap > self.max_allowed_gap_minutes:
                continue
            
            # Determine how many rows to insert
            num_insertions = int(time_gap / self.oversample_interval_minutes) - 1
            if num_insertions <= 0:
                continue
            
            # Generate intermediate timestamps
            timestamps = [
                row1["Datetime"] + timedelta(minutes=(j + 1) * self.oversample_interval_minutes) 
                for j in range(num_insertions)
            ]
            
            # Columns to interpolate
            cols = [col for col in df.columns if col != "Datetime"]
            value_sequences = {}
            
            for col in cols:
                start = row1[col]
                end = row2[col]
                
                if start < end:
                    seq = np.sort(np.random.uniform(start, end, num_insertions))
                elif start > end:
                    seq = np.sort(np.random.uniform(end, start, num_insertions))[::-1]
                else:
                    seq = np.full(num_insertions, start)
                
                value_sequences[col] = seq
            
            # Create interpolated rows
            for j in range(num_insertions):
                new_row = {col: value_sequences[col][j] for col in cols}
                new_row["Datetime"] = timestamps[j]
                oversampled_rows.append(pd.Series(new_row))
        
        # Always include the last row
        oversampled_rows.append(df.iloc[-1])
        
        # Create and return final DataFrame
        oversampled_df = pd.DataFrame(oversampled_rows)
        oversampled_df = oversampled_df.sort_values("Datetime").reset_index(drop=True)
        
        return oversampled_df
    
    def get_preprocessing_summary(self, original_df, processed_df, dataset_type):
        """Generate preprocessing summary"""
        return {
            'dataset_type': dataset_type,
            'original_rows': len(original_df),
            'processed_rows': len(processed_df),
            'rows_added': len(processed_df) - len(original_df),
            'date_range': f"{processed_df['Datetime'].min()} to {processed_df['Datetime'].max()}",
            'oversampling_method': 'Linear Interpolation' if dataset_type == 'lower' else 'Monotonic Random'
        }

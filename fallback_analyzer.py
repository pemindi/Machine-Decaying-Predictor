import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from datetime import timedelta, datetime
import warnings
import os

warnings.filterwarnings("ignore")

class FallbackAnalyzer:
    def __init__(self, graphs_folder, today):
        self.graphs_folder = graphs_folder
        self.today = today
        self.colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    # === Model Functions ===
    def exp_model(self, x, a, b, c):
        return a * np.exp(b * x) + c
    
    def poly_model(self, x, *coeffs):
        x = np.array(x)
        coeffs = np.array(coeffs)
        powers = np.arange(len(coeffs))
        return np.sum(coeffs * x[:, None]**powers, axis=1)
    
    def fit_model(self, x, y, model_func, p0=None, maxfev=10000):
        try:
            popt, _ = curve_fit(model_func, x, y, p0=p0, maxfev=maxfev)
            y_pred = model_func(x, *popt)
            return popt, y_pred, r2_score(y, y_pred)
        except:
            return None, None, -np.inf
    
    def bootstrap_ci(self, x, y, model_func, popt, num_samples=50, future_days=None):
        preds = []
        n = len(x)
        for _ in range(num_samples):
            idx = np.random.choice(n, n, replace=True)
            try:
                p, _ = curve_fit(model_func, x[idx], y[idx], p0=popt, maxfev=10000)
                preds.append(model_func(future_days, *p))
            except:
                continue
        if preds:
            preds = np.array(preds)
            return np.percentile(preds, 5, axis=0), np.percentile(preds, 95, axis=0)
        else:
            return np.full_like(future_days, np.nan), np.full_like(future_days, np.nan)
    
    def compute_rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def analyze_csv_sensor(self, df, parameters=['RMS'], data_percentage=0.25, 
                          curve_fit_percentage=0.75, fall_back_days=1, 
                          start_date=None, end_date=None):
        """Main analysis function - automatically selects best model"""
        
        # Prepare data
        df = df.copy()
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.sort_values('Datetime').reset_index(drop=True)
        
        # Resample to hourly data
        df = df.resample('1H', on='Datetime').mean(numeric_only=True).dropna().reset_index()
        df['time_days'] = (df['Datetime'] - df['Datetime'].min()).dt.total_seconds() / 86400
        
        # Store full dataframe for baseline calculation
        full_df = df.copy()
        
        # If date range is specified, filter the data for modeling
        if start_date and end_date:
            mask = (df['Datetime'] >= pd.to_datetime(start_date)) & (df['Datetime'] <= pd.to_datetime(end_date))
            df = df.loc[mask].copy()
            if len(df) < 10:
                return []
        
        results = []
        
        for param in parameters:
            if param not in df.columns or df[param].nunique() < 10:
                continue
            
            # Calculate baseline from full data (or filtered data if no date range)
            baseline_data = full_df if start_date and end_date else df
            baseline_size = int(len(baseline_data) * data_percentage)
            baseline = baseline_data.iloc[:baseline_size]
            
            mean_val = baseline[param].mean()
            std_val = baseline[param].std()
            threshold = mean_val + 4 * std_val
            
            # Determine fitting region
            if start_date and end_date:
                # Use all selected date range data for fitting
                x_data = df['time_days'].values
                y_data = df[param].values
            else:
                # Use curve_fit_percentage for fitting
                fit_start_idx = int(len(df) * (1 - curve_fit_percentage))
                fit_start_idx = max(fit_start_idx, baseline_size)
                x_data = df['time_days'].values[fit_start_idx:]
                y_data = df[param].values[fit_start_idx:]
            
            # Try all models and select the best one automatically
            model_used, popt, y_model, model_func, best_r2 = None, None, None, None, -np.inf
            
            # Try exponential model first
            popt_exp, _, r2_exp = self.fit_model(x_data, y_data, self.exp_model)
            if r2_exp > 0.7:
                model_used, popt, model_func = "Exponential", popt_exp, self.exp_model
                y_model = self.exp_model(df['time_days'].values, *popt_exp)
                best_r2 = r2_exp
            else:
                # Try polynomial models
                for deg in [2, 3]:
                    poly_func = lambda x, *c: self.poly_model(x, *c)
                    p0 = np.ones(deg + 1)
                    popt_poly, _, r2_poly = self.fit_model(x_data, y_data, poly_func, p0)
                    if r2_poly > best_r2:
                        model_used, popt, model_func = f"Polynomial (deg {deg})", popt_poly, poly_func
                        y_model = model_func(df['time_days'].values, *popt_poly)
                        best_r2 = r2_poly
            
            if popt is None or model_func is None:
                continue
            
            # Generate future predictions
            max_days = 365
            extension = 30
            future_end = x_data.max() + extension
            future_days = np.linspace(x_data.min(), future_end, 300)
            predictions = model_func(future_days, *popt)
            
            # Extend prediction if threshold not reached
            while np.nanmax(predictions) < threshold and future_days[-1] < x_data.min() + max_days:
                future_end += extension
                future_days = np.linspace(x_data.min(), min(future_end, x_data.min() + max_days), 300)
                predictions = model_func(future_days, *popt)
            
            # Find failure point
            fail_idx = np.where(predictions >= threshold)[0]
            failure_day = future_days[fail_idx[0]] if len(fail_idx) > 0 else None
            failure_date = df['Datetime'].min() + timedelta(days=failure_day) if failure_day else None
            
            # Calculate confidence intervals
            ci_lower, ci_upper = self.bootstrap_ci(x_data, y_data, model_func, popt, future_days=future_days)
            
            # Calculate metrics
            rmse = self.compute_rmse(y_data, model_func(x_data, *popt))
            mean_ci_width = np.nanmean(ci_upper - ci_lower)
            
            # Fallback analysis
            fallback_results = self.perform_fallback_analysis(df, param, threshold, fall_back_days)
            
            results.append({
                'param': param,
                'model': model_used,
                'r2': best_r2,
                'rmse': rmse,
                'mean_ci_width': mean_ci_width,
                'mean': mean_val,
                'std': std_val,
                'threshold': threshold,
                'failure_day': failure_day,
                'failure_date': failure_date,
                'future_days': future_days,
                'predictions': predictions,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'y_model': y_model,
                'df': df,
                'x_data': x_data,
                'y_data': y_data,
                'fallback_results': fallback_results,
                'data_percentage': data_percentage,
                'curve_fit_percentage': curve_fit_percentage,
                'fall_back_days': fall_back_days
            })
        
        return results
    
    def perform_fallback_analysis(self, df, param, threshold, fall_back_days):
        """Perform fallback analysis with different time windows - automatically selects best model"""
        fallback_fits = []
        last_date = df['Datetime'].max()
        
        for days in range(fall_back_days, 60 + fall_back_days, fall_back_days):
            start = last_date - timedelta(days=days)
            sub_df = df[df['Datetime'] >= start].copy()
            
            if len(sub_df) < 10:
                continue
            
            x = (sub_df['Datetime'] - df['Datetime'].min()).dt.total_seconds() / 86400
            y = sub_df[param].values
            
            best_r2 = -np.inf
            best_model = None
            best_popt = None
            
            # Try exponential first
            popt, _, r2 = self.fit_model(x, y, self.exp_model)
            if r2 > 0.7 and r2 > best_r2:
                best_model = self.exp_model
                best_popt = popt
                best_r2 = r2
            
            # Try polynomial models
            for deg in [2, 3]:
                p0 = np.ones(deg + 1)
                p_poly, _, r2_poly = self.fit_model(x, y, lambda x, *c: self.poly_model(x, *c), p0)
                if r2_poly > best_r2:
                    best_model = lambda x, *c: self.poly_model(x, *c)
                    best_popt = p_poly
                    best_r2 = r2_poly
            
            if best_model and best_popt is not None:
                future_days = np.linspace(x.min(), x.max() + 30, 200)
                preds = best_model(future_days, *best_popt)
                
                failure_idx = np.where(preds >= threshold)[0]
                fail_day = future_days[failure_idx[0]] if len(failure_idx) else None
                fail_date = df['Datetime'].min() + timedelta(days=fail_day) if fail_day else None
                
                rmse = self.compute_rmse(y, best_model(x, *best_popt))
                
                fallback_fits.append({
                    'x': x, 'y': y, 'model': best_model, 'popt': best_popt, 'r2': best_r2,
                    'rmse': rmse, 'fail_date': fail_date,
                    'failure_predicted': fail_date is not None,
                    'future_days': future_days, 'preds': preds,
                    'range_start': start, 'range_end': last_date,
                    'days_back': days
                })
        
        # Sort by R² and select best future prediction
        fallback_fits.sort(key=lambda f: f['r2'], reverse=True)
        
        today = pd.Timestamp.today().normalize()
        selected = None
        
        for fit in fallback_fits:
            if fit['failure_predicted'] and fit['fail_date'] and fit['fail_date'] > today:
                selected = fit
                break
        
        if not selected and fallback_fits:
            selected = fallback_fits[0]
            if selected:
                selected['fail_date'] = None
                selected['failure_predicted'] = False
        
        return {
            'all_fits': fallback_fits,
            'selected_fit': selected
        }
    
    def create_fallback_visualization(self, result, dataset_name, session_id=""):
        """Create visualization for fallback analysis - shows BEST MODEL only"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        df = result['df']
        param = result['param']
        
        # === Main Analysis Plot ===
        ax1.plot(df['Datetime'], df[param], label=param, color='blue', alpha=0.7)
        
        # Show baseline region
        baseline_end_idx = int(len(df) * result['data_percentage'])
        if baseline_end_idx < len(df):
            baseline_end_time = df['Datetime'].iloc[baseline_end_idx]
            ax1.axvspan(df['Datetime'].min(), baseline_end_time, 
                       color='gray', alpha=0.2, label='Baseline Region')
        
        # Show curve fit region
        if result['curve_fit_percentage'] < 1.0:
            fit_start_idx = int(len(df) * (1 - result['curve_fit_percentage']))
            fit_start_idx = max(fit_start_idx, baseline_end_idx)
            if fit_start_idx < len(df):
                fit_start_time = df['Datetime'].iloc[fit_start_idx]
                ax1.axvspan(fit_start_time, df['Datetime'].max(), 
                           color='orange', alpha=0.1, label='Curve Fit Region')
        
        # Plot BEST model fit
        if result['y_model'] is not None:
            ax1.plot(df['Datetime'], result['y_model'], '--', 
                    label=f"{result['model']} Fit (R²={result['r2']:.3f}) - BEST MODEL", 
                    color='orange', linewidth=2)
        
        # Plot predictions
        ext_dates = df['Datetime'].min() + pd.to_timedelta(result['future_days'], unit='D')
        ax1.plot(ext_dates, result['predictions'], ':', 
                label='Future Prediction', color='green', linewidth=2)
        
        # Plot confidence intervals
        if not np.all(np.isnan(result['ci_lower'])):
            ax1.fill_between(ext_dates, result['ci_lower'], result['ci_upper'], 
                           alpha=0.2, label='90% Confidence Interval', color='green')
        
        # Plot threshold and failure date
        ax1.axhline(result['threshold'], color='red', linestyle='--', 
                   label=f'Failure Threshold ({result["threshold"]:.2f})', linewidth=2)
        
        if result['failure_date']:
            ax1.axvline(result['failure_date'], color='purple', linestyle='--', 
                       label=f"Predicted Failure: {result['failure_date'].date()}", linewidth=2)
        
        # Today's date
        ax1.axvline(self.today, color='black', linestyle='-', 
                   label='Today', alpha=0.8, linewidth=2)
        
        ax1.set_title(f'{dataset_name} - Fallback Analysis - {param} (Best Model: {result["model"]})', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel(param, fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # === Fallback Fit Plot ===
        fallback = result['fallback_results']['selected_fit']
        if fallback:
            # Plot full data in background
            ax2.plot(df['Datetime'], df[param], label=param, alpha=0.3, color='lightblue')
            
            # Plot selected fallback data
            fallback_dates = df['Datetime'].min() + pd.to_timedelta(fallback['x'], unit='D')
            ax2.plot(fallback_dates, fallback['y'], 'o', 
                    label=f'Fallback Data ({fallback["days_back"]} days)', 
                    color='blue', markersize=4)
            
            # Plot fallback fit
            fit_dates = df['Datetime'].min() + pd.to_timedelta(fallback['future_days'], unit='D')
            ax2.plot(fit_dates, fallback['preds'], '--', 
                    label=f"Best Fallback Fit (R²={fallback['r2']:.3f})", 
                    color='orange', linewidth=2)
            
            # Plot threshold and failure prediction
            ax2.axhline(result['threshold'], color='red', linestyle='--', 
                       label='Failure Threshold', linewidth=2)
            
            if fallback['fail_date']:
                ax2.axvline(fallback['fail_date'], color='purple', linestyle='--', 
                           label=f"Fallback Prediction: {fallback['fail_date'].date()}", linewidth=2)
            
            # Today's date
            ax2.axvline(self.today, color='black', linestyle='-', 
                       label='Today', alpha=0.8, linewidth=2)
            
            ax2.set_title(f'Best Fallback Analysis ({fallback["range_start"].date()} to {fallback["range_end"].date()})', 
                         fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No valid fallback fit found', 
                    transform=ax2.transAxes, ha='center', va='center', 
                    fontsize=14, color='red')
            ax2.set_title('Fallback Analysis - No Valid Fit', fontsize=14, fontweight='bold')
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel(param, fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{session_id}_fallback_{dataset_name.lower().replace(' ', '_')}_{param.lower().replace(' ', '_')}.png"
        filepath = os.path.join(self.graphs_folder, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from datetime import timedelta

class ExponentialModel:
    def __init__(self):
        self.model_type = 'exponential'
        self.name = 'Exponential Decay'
    
    def fit(self, x_vals, y_vals):
        """Fit exponential decay model"""
        try:
            # Remove zeros and negative values for log transformation
            valid_mask = y_vals > 0
            if not np.any(valid_mask):
                return None
                
            x_valid = x_vals[valid_mask]
            y_valid = y_vals[valid_mask]
            
            # Fit log(y) = log(a) + b*x
            log_y = np.log(y_valid)
            coeffs = np.polyfit(x_valid, log_y, 1)
            
            # Convert back to exponential form: y = a * exp(b*x)
            b, log_a = coeffs
            a = np.exp(log_a)
            
            # Calculate fitted values for all x
            fitted_y = a * np.exp(b * x_vals)
            
            # Calculate R² and RMSE only for valid points
            fitted_y_valid = a * np.exp(b * x_valid)
            r2 = r2_score(y_valid, fitted_y_valid)
            rmse = np.sqrt(mean_squared_error(y_valid, fitted_y_valid))
            
            return {
                'model': lambda x: a * np.exp(b * x),
                'fitted_y': fitted_y,
                'r2': r2,
                'rmse': rmse,
                'coefficients': [a, b],
                'model_type': self.model_type,
                'name': self.name
            }
        except Exception as e:
            print(f"Error fitting exponential model: {e}")
            return None
    
    def predict_future(self, model_result, x_vals, forecast_days=365):
        """Generate future predictions"""
        if model_result is None:
            return None, None
            
        try:
            last_x = x_vals[-1]
            future_x = np.arange(last_x + 0.1, last_x + forecast_days, 0.1)
            future_y = model_result['model'](future_x)
            return future_x, future_y
        except:
            return None, None
    
    def find_critical_date(self, model_result, x_vals, start_datetime, critical_threshold=10):
        """Find when RUL reaches critical threshold"""
        future_x, future_y = self.predict_future(model_result, x_vals)
        
        if future_x is None or future_y is None:
            return None
            
        try:
            below_critical = np.where(future_y <= critical_threshold)[0]
            if len(below_critical) > 0:
                critical_x = future_x[below_critical[0]]
                critical_date = start_datetime + timedelta(days=critical_x)
                return critical_date
        except:
            pass
            
        return None

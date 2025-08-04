import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from datetime import timedelta

class PolynomialModel:
    def __init__(self, degree):
        self.degree = degree
        self.model_type = f'polynomial_{degree}'
        self.name = f'Polynomial Degree {degree}'
    
    def fit(self, x_vals, y_vals):
        """Fit polynomial regression model"""
        try:
            coeffs = np.polyfit(x_vals, y_vals, deg=self.degree)
            poly = np.poly1d(coeffs)
            
            fitted_y = poly(x_vals)
            r2 = r2_score(y_vals, fitted_y)
            rmse = np.sqrt(mean_squared_error(y_vals, fitted_y))
            
            return {
                'model': poly,
                'fitted_y': fitted_y,
                'r2': r2,
                'rmse': rmse,
                'coefficients': coeffs,
                'model_type': self.model_type,
                'name': self.name
            }
        except Exception as e:
            print(f"Error fitting polynomial model (degree {self.degree}): {e}")
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

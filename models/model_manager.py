import numpy as np
from datetime import timedelta
from .linear_model import LinearModel
from .polynomial_model import PolynomialModel
from .exponential_model import ExponentialModel

class ModelManager:
    def __init__(self):
        self.available_models = {
            'linear': LinearModel(),
            'polynomial_1': PolynomialModel(1),
            'polynomial_2': PolynomialModel(2),
            'polynomial_3': PolynomialModel(3),
            'polynomial_4': PolynomialModel(4),
            'exponential': ExponentialModel()
        }
        
        self.model_names = {
            'linear': 'Linear Regression',
            'polynomial_1': 'Polynomial Degree 1',
            'polynomial_2': 'Polynomial Degree 2',
            'polynomial_3': 'Polynomial Degree 3',
            'polynomial_4': 'Polynomial Degree 4',
            'exponential': 'Exponential Decay'
        }
    
    def get_model(self, model_type):
        """Get model instance by type"""
        return self.available_models.get(model_type)
    
    def get_model_names(self):
        """Get all model names"""
        return self.model_names

    def fit_models(self, x_vals, y_vals, selected_models=None):
        """Fit selected models to data"""
        if selected_models is None:
            selected_models = list(self.available_models.keys())
        
        print(f"Fitting models with x_vals shape: {x_vals.shape}, y_vals shape: {y_vals.shape}")
        
        # Validate input dimensions
        if len(x_vals) != len(y_vals):
            print(f"ERROR in fit_models: x_vals length {len(x_vals)} != y_vals length {len(y_vals)}")
            return {}
        
        if len(x_vals) < 2:
            print(f"ERROR in fit_models: Insufficient data points: {len(x_vals)}")
            return {}
        
        results = {}
        for model_type in selected_models:
            if model_type in self.available_models:
                try:
                    model = self.available_models[model_type]
                    result = model.fit(x_vals, y_vals)
                    results[model_type] = result
                    if result:
                        print(f"  {model_type}: R²={result['r2']:.4f}, RMSE={result['rmse']:.4f}")
                    else:
                        print(f"  {model_type}: Failed to fit")
                except Exception as e:
                    print(f"  {model_type}: Error - {str(e)}")
                    results[model_type] = None
        
        return results

    def predict_critical_date(self, model_result, x_vals, start_datetime, critical_threshold=10):
        """Predict when RUL will reach critical threshold for a single model"""
        if model_result is None:
            return None

        last_x = x_vals[-1]
        future_x = np.arange(last_x + 0.1, last_x + 365, 0.1)

        try:
            if model_result['model_type'] == 'linear':
                future_y = model_result['model'].predict(future_x.reshape(-1, 1))
            elif 'polynomial' in model_result['model_type']:
                future_y = model_result['model'](future_x)
            elif model_result['model_type'] == 'exponential':
                future_y = model_result['model'](future_x)
            else:
                return None

            below_critical = np.where(future_y <= critical_threshold)[0]
            if len(below_critical) > 0:
                critical_x = future_x[below_critical[0]]
                critical_date = start_datetime + timedelta(days=critical_x)
                return critical_date

        except Exception as e:
            print(f"Error predicting critical date for {model_result['model_type']}: {e}")

        return None

    def predict_critical_dates(self, model_results, x_vals, start_datetime, critical_threshold=10):
        """Predict critical dates for all selected models"""
        critical_dates = {}

        for model_type, model_result in model_results.items():
            critical_date = self.predict_critical_date(model_result, x_vals, start_datetime, critical_threshold)
            critical_dates[model_type] = critical_date

        return critical_dates

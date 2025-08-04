import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import os

class Visualizer:
    def __init__(self, graphs_folder, today):
        self.graphs_folder = graphs_folder
        self.today = today
        self.colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    def create_comprehensive_visualization(self, data, model_results, model_manager, 
                                        dataset_name, batch_info, critical_threshold=10, 
                                        session_id=""):
        """Create visualization with forecast lines for all models"""
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()
        
        X = data['Datetime']
        y = data['Predicted RUL'].values
        x_vals = self._datetime_to_elapsed_days(X).values
        
        model_types = ['linear', 'polynomial_1', 'polynomial_2', 
                      'polynomial_3', 'polynomial_4', 'exponential']
        
        for i, model_type in enumerate(model_types):
            ax = axes[i]
            result = model_results.get(model_type)
            
            # Plot actual data
            ax.scatter(X, y, color='lightblue', s=15, alpha=0.7, label='Actual RUL', zorder=3)
            
            if result is not None:
                # Plot fitted line
                ax.plot(X, result['fitted_y'], color=self.colors[i], 
                       linewidth=3, label=f'{result["name"]} Fit', zorder=4)
                
                # Plot forecast line
                model = model_manager.get_model(model_type)
                future_x, future_y = model.predict_future(result, x_vals, forecast_days=180)
                
                if future_x is not None and future_y is not None:
                    # Convert future x back to datetime
                    start_datetime = X.iloc[0]
                    future_dates = [start_datetime + timedelta(days=x) for x in future_x]
                    
                    # Only show reasonable forecast values
                    valid_forecast = future_y > 0
                    if np.any(valid_forecast):
                        ax.plot(np.array(future_dates)[valid_forecast], 
                               future_y[valid_forecast], 
                               color=self.colors[i], linestyle='--', 
                               linewidth=2, alpha=0.8, 
                               label=f'{result["name"]} Forecast', zorder=2)
                
                # Plot critical threshold
                ax.axhline(y=critical_threshold, color='red', linestyle='-', 
                          alpha=0.8, linewidth=2, label=f'Critical Threshold ({critical_threshold})', zorder=1)
                
                # Add today's date marker
                ax.axvline(x=self.today, color='green', linestyle='-', 
                          alpha=0.8, linewidth=2, label='Today', zorder=1)
                
                # Add R² and RMSE to title
                ax.set_title(f'{result["name"]}\nR²: {result["r2"]:.4f}, RMSE: {result["rmse"]:.2f}', 
                           fontsize=14, fontweight='bold')
            else:
                ax.set_title(f'{model_manager.model_names[model_type]}\nFailed to fit', 
                           fontsize=14, color='red')
            
            ax.set_xlabel('Datetime', fontsize=12)
            ax.set_ylabel('Predicted RUL', fontsize=12)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
        
        plt.suptitle(f'{dataset_name} - {batch_info}', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save plot
        filename = f"{session_id}_{dataset_name.lower().replace(' ', '_')}_{batch_info.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}.png"
        filepath = os.path.join(self.graphs_folder, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename
    
    def _datetime_to_elapsed_days(self, dt_series):
        """Convert datetime series to elapsed days from start"""
        start = dt_series.iloc[0]
        return (dt_series - start).dt.total_seconds() / (3600 * 24)

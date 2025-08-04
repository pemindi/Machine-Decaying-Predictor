from data_processor import DataProcessor
from visualizer import Visualizer
from models.model_manager import ModelManager
import os

class MachineDecayPredictor:
    def __init__(self, graphs_folder):
        """Initialize the predictor with only the required graphs folder"""
        self.data_processor = DataProcessor()
        self.model_manager = ModelManager()
        self.visualizer = Visualizer(graphs_folder, self.data_processor.today)
        self.today = self.data_processor.today
    
    def process_dataset(self, data, batch_size, directions=['forward'], 
                       critical_threshold=10, selected_models=None, session_id=""):
        """Process dataset with incremental chunking"""
        if selected_models is None:
            selected_models = list(self.model_manager.available_models.keys())
        
        all_results = {}
        all_graphs = []
        
        for direction in directions:
            print(f"Processing {direction} direction...")
            
            # Create incremental batches
            batches = self.data_processor.create_incremental_batches(
                data, batch_size, direction
            )
            
            batch_results = []
            
            for batch in batches:
                batch_data = batch['data']
                batch_info = batch['info']
                
                print(f"  Processing batch: {batch_info}")
                print(f"  Batch data shape: {batch_data.shape}")
                
                # Ensure we have enough data points
                if len(batch_data) < 2:
                    print(f"  Skipping batch with insufficient data: {len(batch_data)} rows")
                    continue
                
                # Prepare data for modeling - CRITICAL FIX HERE
                X = batch_data['Datetime'].reset_index(drop=True)
                y = batch_data['Predicted RUL'].values
                
                # Double-check dimensions match
                print(f"  X (datetime) length: {len(X)}")
                print(f"  y (RUL) length: {len(y)}")
                
                if len(X) != len(y):
                    print(f"  ERROR: Dimension mismatch! X={len(X)}, y={len(y)}")
                    # Force them to match by taking the minimum length
                    min_len = min(len(X), len(y))
                    X = X.iloc[:min_len]
                    y = y[:min_len]
                    print(f"  Fixed: Both now have length {min_len}")
                
                x_vals = self.data_processor.datetime_to_elapsed_days(X).values
                
                # Final check
                if len(x_vals) != len(y):
                    print(f"  FINAL ERROR: x_vals={len(x_vals)}, y={len(y)}")
                    continue
                
                # Fit models
                model_results = self.model_manager.fit_models(
                    x_vals, y, selected_models
                )
                
                # Predict critical dates
                critical_dates = self.model_manager.predict_critical_dates(
                    model_results, x_vals, X.iloc[0], critical_threshold
                )
                
                # Create visualization
                graph_filename = self.visualizer.create_comprehensive_visualization(
                    batch_data, model_results, self.model_manager,
                    f"Dataset ({direction.title()})", batch_info,
                    critical_threshold, session_id
                )
                all_graphs.append(graph_filename)
                
                batch_results.append({
                    'batch_size': batch['size'],
                    'date_range': batch['date_range'],
                    'model_results': model_results,
                    'critical_dates': critical_dates,
                    'batch_data': batch_data,
                    'batch_info': batch_info,
                    'graph_filename': graph_filename
                })
            
            all_results[direction] = batch_results
        
        return all_results, all_graphs

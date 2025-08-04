from flask import Blueprint, request, redirect, url_for, flash, render_template, current_app
import pandas as pd
import uuid
from datetime import datetime
from utils.file_utils import allowed_file, find_session_files, save_uploaded_files
from utils.report_generator import create_excel_report, create_fallback_excel_report
from predictor import MachineDecayPredictor
from data_preprocessor import DataPreprocessor
from fallback_analyzer import FallbackAnalyzer
import os

analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/analyze_incremental', methods=['POST'])
def analyze_incremental():
    """Handle incremental analysis requests"""
    try:
        # Check if this is a retry session
        retry_session_id = request.form.get('retry_session_id')
        
        if retry_session_id:
            # Use existing session ID and files
            session_id = retry_session_id
            session_files = find_session_files(session_id, current_app.config['UPLOAD_FOLDER'])
            
            if not session_files:
                flash('Previous session files not found. Please upload new files.', 'error')
                return redirect(url_for('main.index'))
            
            lower_path = session_files['lower_path']
            upper_path = session_files['upper_path']
            
            print(f"Using retry session {session_id} with existing files:")
            print(f"  - Lower: {session_files['lower_name']}")
            print(f"  - Upper: {session_files['upper_name']}")
        else:
            # Generate new session ID
            session_id = str(uuid.uuid4())[:8]
            
            # Get uploaded files
            lower_file = request.files.get('lower_file')
            upper_file = request.files.get('upper_file')
            
            if not lower_file or not upper_file:
                flash('Please upload both CSV files', 'error')
                return redirect(url_for('main.index'))
            
            if not (allowed_file(lower_file.filename) and allowed_file(upper_file.filename)):
                flash('Please upload valid CSV files', 'error')
                return redirect(url_for('main.index'))
            
            # Save uploaded files
            lower_path, upper_path = save_uploaded_files(
                lower_file, upper_file, session_id, current_app.config['UPLOAD_FOLDER']
            )
            
            print(f"New incremental analysis session {session_id}")
        
        # Get current analysis date
        analysis_date = datetime.now()
        
        # Get form data
        batch_size = int(request.form.get('batch_size', 100))
        directions = request.form.getlist('direction')
        if not directions:
            directions = ['forward']
        
        critical_threshold = float(request.form.get('critical_threshold', 10))
        selected_models = request.form.getlist('models')
        if not selected_models:
            selected_models = ['linear', 'polynomial_1', 'polynomial_2', 'polynomial_3', 'polynomial_4', 'exponential']
        
        # Check if preprocessing should be applied
        apply_preprocessing = request.form.get('apply_preprocessing') == 'on'
        
        print(f"Incremental analysis settings:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Directions: {directions}")
        print(f"  - Critical threshold: {critical_threshold}")
        print(f"  - Selected models: {selected_models}")
        print(f"  - Apply preprocessing: {apply_preprocessing}")
        
        # Read original data
        lower_df_original = pd.read_csv(lower_path)
        upper_df_original = pd.read_csv(upper_path)
        
        print(f"Original data shapes: Lower={lower_df_original.shape}, Upper={upper_df_original.shape}")
        
        preprocessing_summaries = []
        
        # Apply preprocessing if requested
        if apply_preprocessing:
            print("Applying preprocessing...")
            preprocessor = DataPreprocessor()
            
            # Process lower data
            lower_df_processed = preprocessor.process_lower_data(lower_df_original)
            lower_summary = preprocessor.get_preprocessing_summary(
                lower_df_original, lower_df_processed, 'lower'
            )
            preprocessing_summaries.append(lower_summary)
            
            # Process upper data
            upper_df_processed = preprocessor.process_upper_data(upper_df_original)
            upper_summary = preprocessor.get_preprocessing_summary(
                upper_df_original, upper_df_processed, 'upper'
            )
            preprocessing_summaries.append(upper_summary)
            
            # Use processed data for analysis
            lower_df = lower_df_processed
            upper_df = upper_df_processed
            
            flash(f'Preprocessing applied: Lower {lower_summary["original_rows"]} → {lower_summary["processed_rows"]} rows, Upper {upper_summary["original_rows"]} → {upper_summary["processed_rows"]} rows', 'success')
        else:
            print("Skipping preprocessing - using original data")
            # Use original data
            lower_df = lower_df_original
            upper_df = upper_df_original
        
        # Initialize predictor
        print("Initializing predictor...")
        predictor = MachineDecayPredictor(current_app.config['GRAPHS_FOLDER'])
        
        # Prepare data with minimal filtering when preprocessing is disabled
        minimal_filtering = not apply_preprocessing
        print(f"Preparing data with minimal_filtering={minimal_filtering}")
        
        lower_data = predictor.data_processor.prepare_data(lower_df, apply_minimal_filtering=minimal_filtering)
        upper_data = predictor.data_processor.prepare_data(upper_df, apply_minimal_filtering=minimal_filtering)
        
        print(f"Prepared data shapes: Lower={lower_data.shape}, Upper={upper_data.shape}")
        
        # Process datasets with incremental chunking
        print("Processing lower dataset...")
        lower_results, lower_graphs = predictor.process_dataset(
            lower_data, batch_size, directions, critical_threshold, selected_models, session_id
        )
        
        print("Processing upper dataset...")
        upper_results, upper_graphs = predictor.process_dataset(
            upper_data, batch_size, directions, critical_threshold, selected_models, session_id
        )
        
        # Combine all graphs
        all_graphs = lower_graphs + upper_graphs
        print(f"Generated {len(all_graphs)} visualizations")
        
        # Create Excel report with current analysis date
        print("Creating Excel report...")
        report_path = create_excel_report(
            lower_results, upper_results, critical_threshold, session_id, 
            predictor.model_manager, preprocessing_summaries if apply_preprocessing else None, 
            analysis_date, current_app.config['REPORTS_FOLDER']
        )
        
        # Prepare results for template
        results_data = {
            'lower_results': lower_results,
            'upper_results': upper_results,
            'graph_files': all_graphs,
            'report_filename': os.path.basename(report_path),
            'session_id': session_id,
            'batch_size': batch_size,
            'directions': directions,
            'selected_models': selected_models,
            'available_models': predictor.model_manager.get_model_names(),
            'critical_threshold': critical_threshold,
            'today': predictor.today,
            'preprocessing_applied': apply_preprocessing,
            'preprocessing_summaries': preprocessing_summaries,
            'analysis_date': analysis_date,
            'method_type': 'incremental'
        }
        
        print("Incremental analysis completed successfully!")
        return render_template('results.html', **results_data)
        
    except Exception as e:
        print(f"Error during incremental analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error during incremental analysis: {str(e)}', 'error')
        return redirect(url_for('main.index'))

@analysis_bp.route('/analyze_fallback', methods=['POST'])
def analyze_fallback():
    """Handle fallback analysis requests"""
    try:
        # Check if this is a retry session
        retry_session_id = request.form.get('retry_session_id')
        
        if retry_session_id:
            # Use existing session ID and files
            session_id = retry_session_id
            session_files = find_session_files(session_id, current_app.config['UPLOAD_FOLDER'])
            
            if not session_files:
                flash('Previous session files not found. Please upload new files.', 'error')
                return redirect(url_for('main.index'))
            
            lower_path = session_files['lower_path']
            upper_path = session_files['upper_path']
            
            print(f"Using retry session {session_id} with existing files:")
            print(f"  - Lower: {session_files['lower_name']}")
            print(f"  - Upper: {session_files['upper_name']}")
        else:
            # Generate new session ID
            session_id = str(uuid.uuid4())[:8]
            
            # Get uploaded files
            lower_file = request.files.get('lower_file')
            upper_file = request.files.get('upper_file')
            
            if not lower_file or not upper_file:
                flash('Please upload both CSV files', 'error')
                return redirect(url_for('main.index'))
            
            if not (allowed_file(lower_file.filename) and allowed_file(upper_file.filename)):
                flash('Please upload valid CSV files', 'error')
                return redirect(url_for('main.index'))
            
            # Save uploaded files
            lower_path, upper_path = save_uploaded_files(
                lower_file, upper_file, session_id, current_app.config['UPLOAD_FOLDER']
            )
            
            print(f"New fallback analysis session {session_id}")
        
        # Get current analysis date
        analysis_date = datetime.now()
        
        # Get form data
        fallback_days = int(request.form.get('fallback_days', 1))
        baseline_percentage = float(request.form.get('baseline_percentage', 25)) / 100.0
        curve_fit_percentage = float(request.form.get('curve_fit_percentage', 75)) / 100.0
        parameters = request.form.getlist('parameters')
        if not parameters:
            parameters = ['RMS']
        
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        
        print(f"Fallback analysis settings:")
        print(f"  - Fallback days: {fallback_days}")
        print(f"  - Baseline percentage: {baseline_percentage}")
        print(f"  - Curve fit percentage: {curve_fit_percentage}")
        print(f"  - Parameters: {parameters}")
        print(f"  - Date range: {start_date} to {end_date}")
        
        # Read data
        lower_df = pd.read_csv(lower_path)
        upper_df = pd.read_csv(upper_path)
        
        print(f"Data shapes: Lower={lower_df.shape}, Upper={upper_df.shape}")
        
        # Initialize fallback analyzer
        fallback_analyzer = FallbackAnalyzer(current_app.config['GRAPHS_FOLDER'], analysis_date)
        
        # Analyze both datasets - automatically selects best model
        lower_results = fallback_analyzer.analyze_csv_sensor(
            lower_df, parameters=parameters, 
            data_percentage=baseline_percentage,
            curve_fit_percentage=curve_fit_percentage,
            fall_back_days=fallback_days,
            start_date=start_date,
            end_date=end_date
        )
        
        upper_results = fallback_analyzer.analyze_csv_sensor(
            upper_df, parameters=parameters,
            data_percentage=baseline_percentage,
            curve_fit_percentage=curve_fit_percentage,
            fall_back_days=fallback_days,
            start_date=start_date,
            end_date=end_date
        )
        
        # Generate visualizations - only for best models
        all_graphs = []
        
        for result in lower_results:
            graph_filename = fallback_analyzer.create_fallback_visualization(
                result, "Lower Dataset", session_id
            )
            all_graphs.append(graph_filename)
        
        for result in upper_results:
            graph_filename = fallback_analyzer.create_fallback_visualization(
                result, "Upper Dataset", session_id
            )
            all_graphs.append(graph_filename)
        
        print(f"Generated {len(all_graphs)} fallback visualizations (best models only)")
        
        # Create Excel report
        report_path = create_fallback_excel_report(
            lower_results, upper_results, session_id, analysis_date,
            fallback_days, baseline_percentage, curve_fit_percentage, parameters,
            current_app.config['REPORTS_FOLDER']
        )
        
        # Prepare results for template
        results_data = {
            'lower_results': lower_results,
            'upper_results': upper_results,
            'graph_files': all_graphs,
            'report_filename': os.path.basename(report_path),
            'session_id': session_id,
            'fallback_days': fallback_days,
            'baseline_percentage': baseline_percentage * 100,
            'curve_fit_percentage': curve_fit_percentage * 100,
            'parameters': parameters,
            'start_date': start_date,
            'end_date': end_date,
            'analysis_date': analysis_date,
            'method_type': 'fallback'
        }
        
        print("Fallback analysis completed successfully!")
        return render_template('fallback_results.html', **results_data)
        
    except Exception as e:
        print(f"Error during fallback analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error during fallback analysis: {str(e)}', 'error')
        return redirect(url_for('main.index'))

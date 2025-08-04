from flask import Flask
import os
import warnings
from routes.main_routes import main_bp
from routes.analysis_routes import analysis_bp
from routes.report_routes import report_bp

warnings.filterwarnings('ignore')

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
    
    # Create necessary directories
    UPLOAD_FOLDER = 'uploads'
    REPORTS_FOLDER = 'reports'
    GRAPHS_FOLDER = 'static/graphs'
    
    for folder in [UPLOAD_FOLDER, REPORTS_FOLDER, GRAPHS_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['REPORTS_FOLDER'] = REPORTS_FOLDER
    app.config['GRAPHS_FOLDER'] = GRAPHS_FOLDER
    
    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(report_bp)
    
    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

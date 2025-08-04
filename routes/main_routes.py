from flask import Blueprint, render_template, request
from datetime import datetime
from utils.file_utils import find_session_files

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Main index page with file upload and method selection"""
    # Get current date for display
    current_date = datetime.now().strftime('%B %d, %Y')
    
    # Check if this is a retry session
    try_incremental = request.args.get('try_incremental')
    try_fallback = request.args.get('try_fallback')
    session_id = request.args.get('session')
    
    session_files = None
    if session_id and (try_incremental or try_fallback):
        from flask import current_app
        session_files = find_session_files(session_id, current_app.config['UPLOAD_FOLDER'])
        if session_files:
            method = 'incremental' if try_incremental else 'fallback'
            from flask import flash
            flash(f'Retry mode: Using your previous files to try the {method.title()} method.', 'info')
        else:
            from flask import flash
            flash('Previous session files not found. Please upload new files.', 'warning')
    
    return render_template('index.html', 
                         current_date=current_date,
                         session_files=session_files,
                         try_incremental=try_incremental,
                         try_fallback=try_fallback,
                         session_id=session_id)

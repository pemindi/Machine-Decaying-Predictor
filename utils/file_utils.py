import os
import glob
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_session_files(session_id, upload_folder):
    """Find existing files for a session"""
    try:
        lower_pattern = f"{session_id}_lower_*.csv"
        upper_pattern = f"{session_id}_upper_*.csv"
        
        lower_files = glob.glob(os.path.join(upload_folder, lower_pattern))
        upper_files = glob.glob(os.path.join(upload_folder, upper_pattern))
        
        if lower_files and upper_files:
            return {
                'lower_path': lower_files[0],
                'upper_path': upper_files[0],
                'lower_name': os.path.basename(lower_files[0]),
                'upper_name': os.path.basename(upper_files[0])
            }
        return None
    except Exception as e:
        print(f"Error finding session files: {e}")
        return None

def save_uploaded_files(lower_file, upper_file, session_id, upload_folder):
    """Save uploaded files with session ID prefix"""
    lower_filename = secure_filename(f"{session_id}_lower_{lower_file.filename}")
    upper_filename = secure_filename(f"{session_id}_upper_{upper_file.filename}")
    
    lower_path = os.path.join(upload_folder, lower_filename)
    upper_path = os.path.join(upload_folder, upper_filename)
    
    lower_file.save(lower_path)
    upper_file.save(upper_path)
    
    return lower_path, upper_path

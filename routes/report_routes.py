from flask import Blueprint, send_file, redirect, url_for, flash, current_app
import os

report_bp = Blueprint('report', __name__)

@report_bp.route('/download_report/<filename>')
def download_report(filename):
    """Handle report download requests"""
    try:
        filepath = os.path.join(current_app.config['REPORTS_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            flash('Report file not found', 'error')
            return redirect(url_for('main.index'))
    except Exception as e:
        flash(f'Error downloading report: {str(e)}', 'error')
        return redirect(url_for('main.index'))

#!/usr/bin/env python3
"""
Cleanup script to remove old files from uploads, reports, and graphs directories.
This helps manage disk space when processing many analyses.
"""

import os
import time
from datetime import datetime, timedelta

def cleanup_old_files(directory, days_old=7):
    """Remove files older than specified days."""
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    cutoff_time = time.time() - (days_old * 24 * 60 * 60)
    removed_count = 0
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if os.path.isfile(filepath):
            file_modified_time = os.path.getmtime(filepath)
            
            if file_modified_time < cutoff_time:
                try:
                    os.remove(filepath)
                    print(f"🗑️  Removed: {filename}")
                    removed_count += 1
                except Exception as e:
                    print(f"❌ Error removing {filename}: {e}")
    
    print(f"✅ Removed {removed_count} files from {directory}")

def main():
    """Main cleanup function."""
    
    print("🧹 Starting cleanup of old files...")
    print(f"📅 Removing files older than 7 days")
    print(f"🕒 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    # Cleanup directories
    directories_to_clean = [
        'uploads',
        'reports',
        'static/graphs'
    ]
    
    for directory in directories_to_clean:
        print(f"\n📂 Cleaning directory: {directory}")
        cleanup_old_files(directory, days_old=7)
    
    print("\n🎉 Cleanup completed!")

if __name__ == "__main__":
    main()

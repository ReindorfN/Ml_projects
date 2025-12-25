#!/usr/bin/env python3
"""
Script to download the model file from cloud storage during deployment.
This script will be run during the Render build process.

Supports:
- Google Drive (direct download links)
- Dropbox (with ?dl=1)
- Direct HTTP/HTTPS URLs
"""

import os
import sys
import urllib.request
import urllib.error
import re

def extract_google_drive_file_id(url):
    """Extract file ID from Google Drive shareable link"""
    # Pattern: https://drive.google.com/file/d/FILE_ID/view
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return None

def get_google_drive_download_url(shareable_url):
    """Convert Google Drive shareable link to direct download URL"""
    file_id = extract_google_drive_file_id(shareable_url)
    if file_id:
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return None

def download_model_from_url(url, output_path='rf_logistics_model.pkl'):
    """
    Download model file from a URL (Google Drive, Dropbox, or direct link)
    """
    print(f"Downloading model from {url}...")
    
    # Handle Google Drive links
    if 'drive.google.com' in url and '/file/d/' in url:
        print("Detected Google Drive link, converting to direct download...")
        direct_url = get_google_drive_download_url(url)
        if direct_url:
            url = direct_url
            print(f"Using direct download URL: {url}")
        else:
            print("Warning: Could not extract Google Drive file ID")
    
    # Handle Dropbox links - ensure dl=1
    if 'dropbox.com' in url:
        if '?dl=0' in url:
            url = url.replace('?dl=0', '?dl=1')
        elif '?dl=1' not in url:
            url = url + ('&' if '?' in url else '?') + 'dl=1'
        print(f"Using Dropbox direct download URL: {url}")
    
    try:
        # Create a request with headers to avoid Google Drive virus scan warning
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        # Download with progress
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
            print(f"\rDownloading: {percent:.1f}% ({downloaded / (1024*1024):.2f} MB)", end='', flush=True)
        
        # Download the file
        urllib.request.urlretrieve(req, output_path, reporthook=show_progress)
        print()  # New line after progress
        
        if not os.path.exists(output_path):
            print("✗ Download completed but file not found!")
            return False
        
        file_size = os.path.getsize(output_path)
        file_size_mb = file_size / (1024 * 1024)  # Size in MB
        
        # Check if file is suspiciously small (might be an HTML error page)
        if file_size < 1024:  # Less than 1 KB
            print(f"✗ Downloaded file is too small ({file_size} bytes) - likely an error page")
            # Try to read first few bytes to check if it's HTML
            with open(output_path, 'rb') as f:
                first_bytes = f.read(100)
                if b'<html' in first_bytes.lower() or b'<!doctype' in first_bytes.lower():
                    print("  Detected HTML content - download likely failed (Google Drive virus scan?)")
                    print("  Try: Share the file with 'Anyone with the link' and use a direct download link")
            os.remove(output_path)
            return False
        
        # Check if file looks like HTML (Google Drive sometimes returns HTML for large files)
        with open(output_path, 'rb') as f:
            first_bytes = f.read(500)
            if b'<html' in first_bytes.lower() or b'<!doctype' in first_bytes.lower():
                print(f"✗ Downloaded file appears to be HTML (not a pickle file)")
                print("  This usually means Google Drive is blocking the download")
                print("  Solutions:")
                print("  1. Share file with 'Anyone with the link'")
                print("  2. For large files, use Dropbox or another service")
                os.remove(output_path)
                return False
        
        print(f"✓ Model downloaded successfully! Size: {file_size_mb:.2f} MB ({file_size} bytes)")
        print(f"Downloaded file name: {output_path}")
        
        # Verify it's a valid pickle file by trying to peek at it
        try:
            import pickle
            with open(output_path, 'rb') as f:
                # Just check if we can read the pickle protocol
                pickle_protocol = f.read(1)
                if not pickle_protocol or pickle_protocol[0] not in [0, 1, 2, 3, 4, 5]:
                    print("⚠ Warning: File might not be a valid pickle file")
        except Exception as e:
            print(f"⚠ Warning: Could not verify pickle file format: {e}")
        
        return True
            
    except urllib.error.HTTPError as e:
        print(f"\n✗ HTTP Error {e.code}: {e.reason}")
        if e.code == 403:
            print("  Tip: Make sure the file is shared publicly (Anyone with the link)")
        elif e.code == 404:
            print("  Tip: Check that the file ID in the URL is correct")
        return False
    except urllib.error.URLError as e:
        print(f"\n✗ URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Get URL from environment variable or command line argument
    model_url = os.environ.get('MODEL_DOWNLOAD_URL')
    
    if not model_url:
        if len(sys.argv) > 1:
            model_url = sys.argv[1]
        else:
            print("=" * 60)
            print("Model Download Script")
            print("=" * 60)
            print("\nError: MODEL_DOWNLOAD_URL environment variable not set")
            print("\nUsage options:")
            print("  1. Set environment variable: export MODEL_DOWNLOAD_URL='your_url'")
            print("  2. Pass as argument: python download_model.py <url>")
            print("\nSupported services:")
            print("  - Google Drive (shareable links)")
            print("  - Dropbox (shareable links)")
            print("  - Direct HTTP/HTTPS URLs")
            print("=" * 60)
            sys.exit(1)
    
    if download_model_from_url(model_url):
        print("✓ Model file ready for deployment!")
        sys.exit(0)
    else:
        print("✗ Failed to download model file!")
        print("\nTroubleshooting:")
        print("  - Check that the URL is accessible")
        print("  - For Google Drive: Ensure file is shared with 'Anyone with the link'")
        print("  - For Dropbox: Ensure link has ?dl=1 parameter")
        sys.exit(1)


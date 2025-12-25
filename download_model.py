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
import http.cookiejar
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

def handle_google_drive_download(url, output_path):
    """
    Handle Google Drive downloads, including virus scan warning pages for large files.
    Returns True if successful, False otherwise.
    """
    print("Handling Google Drive download (may need to bypass virus scan warning)...")
    
    # Extract file ID from URL
    file_id_match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url)
    if not file_id_match:
        print("Could not extract file ID from Google Drive URL")
        return False
    file_id = file_id_match.group(1)
    
    # Create opener with cookie handling for Google Drive
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
    urllib.request.install_opener(opener)
    
    try:
        # First, try to download - might get the warning page
        print("Attempting initial download...")
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        response = urllib.request.urlopen(req)
        
        # Read only first 10KB to check if it's the warning page
        content_preview = response.read(10240)
        response.close()
        
        # Check if we got the virus scan warning page
        content_str = content_preview.decode('utf-8', errors='ignore')
        if 'virus scan' in content_str.lower() or 'too large' in content_str.lower():
            print("Detected Google Drive virus scan warning page, extracting download link...")
            
            # Method 1: Look for the confirm token in the form action or link
            confirm_match = re.search(r'confirm=([a-zA-Z0-9_-]+)', content_str)
            if confirm_match:
                confirm_token = confirm_match.group(1)
                actual_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                print(f"Found confirm token, using direct download URL...")
                url = actual_url
            else:
                # Method 2: Try to find the download link in the page
                download_link_match = re.search(r'href="(/uc\?export=download[^"]+)"', content_str)
                if download_link_match:
                    download_path = download_link_match.group(1)
                    url = f"https://drive.google.com{download_path}"
                    print(f"Found download link in page, using: {url}")
                else:
                    # Method 3: Use confirm=t parameter (sometimes works for large files)
                    print("Could not find confirm token, trying confirm=t parameter...")
                    url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
        
        # Now download the actual file
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
            print(f"\rDownloading: {percent:.1f}% ({downloaded / (1024*1024):.2f} MB)", end='', flush=True)
        
        # urlretrieve expects a URL string, not a Request object
        urllib.request.urlretrieve(url, output_path, reporthook=show_progress)
        print()  # New line after progress
        return True
        
    except urllib.error.HTTPError as e:
        print(f"\n✗ HTTP Error {e.code}: {e.reason}")
        if e.code == 403:
            print("  Tip: Make sure the file is shared publicly (Anyone with the link)")
        return False
    except Exception as e:
        print(f"Error during Google Drive download: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_model_from_url(url, output_path='rf_logistics_model.pkl'):
    """
    Download model file from a URL (Google Drive, Dropbox, or direct link)
    """
    print(f"Downloading model from {url}")
    
    is_google_drive = False
    original_url = url
    
    # Handle Google Drive links
    if 'drive.google.com' in url:
        is_google_drive = True
        if '/file/d/' in url:
            print("Detected Google Drive shareable link, converting to direct download...")
            direct_url = get_google_drive_download_url(url)
            if direct_url:
                url = direct_url
                print(f"Using direct download URL: {url}")
            else:
                print("Warning: Could not extract Google Drive file ID")
        # Use special handler for Google Drive
        if handle_google_drive_download(url, output_path):
            # Continue with validation below
            pass
        else:
            return False
    
    # Handle Dropbox links - ensure dl=1
    elif 'dropbox.com' in url:
        if '?dl=0' in url:
            url = url.replace('?dl=0', '?dl=1')
        elif '?dl=1' not in url:
            url = url + ('&' if '?' in url else '?') + 'dl=1'
        print(f"Using Dropbox direct download URL: {url}")
        
        try:
            # Download with progress
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
                print(f"\rDownloading: {percent:.1f}% ({downloaded / (1024*1024):.2f} MB)", end='', flush=True)
            
            # urlretrieve expects a URL string, not a Request object
            urllib.request.urlretrieve(url, output_path, reporthook=show_progress)
            print()  # New line after progress
        except Exception as e:
            print(f"\n✗ Download error: {type(e).__name__}: {e}")
            return False
    
    # Handle direct HTTP/HTTPS URLs
    else:
        try:
            # Create a request with headers
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            # Download with progress
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
                print(f"\rDownloading: {percent:.1f}% ({downloaded / (1024*1024):.2f} MB)", end='', flush=True)
            
            # For direct URLs, we can use urlopen + manual writing for better control
            response = urllib.request.urlopen(req)
            total_size = int(response.headers.get('Content-Length', 0))
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = min(downloaded * 100 / total_size, 100)
                        print(f"\rDownloading: {percent:.1f}% ({downloaded / (1024*1024):.2f} MB)", end='', flush=True)
            print()  # New line after progress
        except Exception as e:
            print(f"\n✗ Download error: {type(e).__name__}: {e}")
            return False
    
    # Validate downloaded file
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


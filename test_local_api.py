import requests
import os
import sys
import argparse
from pathlib import Path

def test_api_status(base_url):
    """Test the API status endpoint"""
    url = f"{base_url}/api/status"
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error accessing status endpoint: {e}")
        return False

def test_verification(base_url, image_path):
    """Test the image verification endpoint"""
    url = f"{base_url}/verify-qr"
    
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as img:
            files = {'qrImage': (os.path.basename(image_path), img, 'image/jpeg')}
            response = requests.post(url, files=files)
            
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing verification endpoint: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test the QR verification API')
    parser.add_argument('--url', default='http://localhost:5001', help='Base URL of the API')
    parser.add_argument('--image', help='Path to an image file to test verification')
    
    args = parser.parse_args()
    
    print(f"Testing API at {args.url}")
    
    status_ok = test_api_status(args.url)
    print(f"Status endpoint test {'passed' if status_ok else 'failed'}")
    
    if args.image:
        verification_ok = test_verification(args.url, args.image)
        print(f"Verification endpoint test {'passed' if verification_ok else 'failed'}")
    else:
        print("Skipping verification test (no image provided)")
        print("To test verification, use --image path/to/qr_image.jpg")

if __name__ == '__main__':
    main() 
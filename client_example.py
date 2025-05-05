import requests
import argparse
import os

def verify_image(api_url, image_path):
    """
    Send an image to the API for verification
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    # Prepare the file for upload
    files = {'image': open(image_path, 'rb')}
    
    try:
        # Send the request to the API
        response = requests.post(f"{api_url}/api/verify", files=files)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return None

def check_api_status(api_url):
    """
    Check if the API is running
    """
    try:
        response = requests.get(f"{api_url}/api/status")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: API returned status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDP Verification Client")
    parser.add_argument("--api_url", default="http://localhost:5000", help="URL of the API server")
    parser.add_argument("--image", required=True, help="Path to the image to verify")
    
    args = parser.parse_args()
    
    # Check API status
    print("Checking API status...")
    status = check_api_status(args.api_url)
    if not status:
        print("API is not available. Please ensure the server is running.")
        exit(1)
    
    print(f"API Status: {status}")
    
    if not status.get("model_loaded", False):
        print("Warning: Model is not loaded on the server.")
    
    # Verify image
    print(f"\nVerifying image: {args.image}")
    result = verify_image(args.api_url, args.image)
    
    if result:
        if "error" in result:
            print(f"Error processing image: {result['error']}")
        else:
            print("\n=== Verification Result ===")
            print(f"Authentic: {'Yes' if result.get('is_authentic', False) else 'No'}")
            print(f"Confidence: {result.get('confidence', 0) * 100:.2f}%")
            
            if "class" in result:
                classes = ["Original", "Fake Type 1", "Fake Type 2", "Fake Type 3", "Fake Type 4"]
                print(f"Class: {classes[result['class']]}")
            
            if "predictions" in result:
                print("\nDetailed predictions:")
                predictions = result["predictions"]
                for i, pred in enumerate(predictions):
                    if i == 0:
                        print(f"  Original: {pred * 100:.2f}%")
                    else:
                        if len(predictions) == 2:
                            print(f"  Fake: {pred * 100:.2f}%")
                        else:
                            print(f"  Fake Type {i}: {pred * 100:.2f}%")
    else:
        print("Failed to get a response from the API.") 
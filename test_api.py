import requests
import time
import subprocess
import os
import sys

def test_api():
    img_path = "test_apple.jpg"
    
    # Generate a dummy test image
    import numpy as np
    import cv2
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:] = (0, 0, 255) # Red color
    cv2.imwrite(img_path, img)

    # Start the FastAPI server in the background
    print("Starting FastAPI server...")
    server_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", "8000"], 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for the server to boot up
    print("Waiting for server to load models (15 seconds)...")
    time.sleep(15) 
    
    # Check if process is still running
    if server_process.poll() is not None:
        print("Server failed to start. Error:")
        print(server_process.stderr.read().decode('utf-8'))
        return

    # Test the API
    api_url = "http://localhost:8000/predict"
    print(f"Testing API at {api_url}...")
    
    try:
        with open(img_path, "rb") as f:
            files = {"file": ("test_apple.jpg", f, "image/jpeg")}
            resp = requests.post(api_url, files=files)
            
        print("Status Code:", resp.status_code)
        try:
            print("Response JSON:", resp.json())
        except:
            print("Response text:", resp.text)
    except Exception as e:
        print(f"API request failed: {e}")
        print("Server stderr:")
        print(server_process.stderr.read().decode('utf-8', errors='ignore'))
        
    finally:
        # Clean up
        print("Killing server...")
        server_process.terminate()
        server_process.wait()
        
        if os.path.exists(img_path):
            os.remove(img_path)
            
if __name__ == "__main__":
    test_api()

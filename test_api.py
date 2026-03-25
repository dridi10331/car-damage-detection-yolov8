"""
Test script for Car Damage Detection API
"""
import requests
import json
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200

def test_root():
    """Test root endpoint"""
    print("Testing / endpoint...")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    print("Testing /model/info endpoint...")
    response = requests.get(f"{API_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200

def test_predict():
    """Test predict endpoint"""
    print("Testing /predict endpoint...")
    
    # Find a test image
    test_images = list(Path("test_results").glob("*.png"))
    if not test_images:
        print("No test images found!")
        return False
    
    test_image = test_images[0]
    print(f"Using test image: {test_image}")
    
    with open(test_image, "rb") as f:
        files = {"file": (test_image.name, f, "image/png")}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Total damages: {result['summary']['total_damages']}")
        print(f"Critical damages: {result['summary']['by_severity']['CRITICAL']}")
        print(f"\nFull response:")
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def main():
    """Run all tests"""
    print("="*60)
    print("Car Damage Detection API - Test Suite")
    print("="*60 + "\n")
    
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Model Info", test_model_info),
        ("Prediction", test_predict),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"Error in {name}: {str(e)}\n")
            results.append((name, False))
    
    print("="*60)
    print("Test Results:")
    print("="*60)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    print("="*60)

if __name__ == "__main__":
    main()

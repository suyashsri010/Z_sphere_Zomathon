"""Quick API test script."""
import requests
import json

# Test data
test_request = {
    "user_id": "user_0",
    "restaurant_id": "rest_0",
    "cart_items": [
        {
            "item_id": "item_0",
            "name": "Japanese Soup",
            "category": "appetizer",
            "price": 176.88,
            "quantity": 1
        }
    ],
    "top_n": 5
}

# Test health endpoint
print("Testing health endpoint...")
response = requests.get("http://127.0.0.1:8000/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}\n")

# Test recommendations endpoint
print("Testing recommendations endpoint...")
response = requests.post(
    "http://127.0.0.1:8000/recommend",
    json=test_request
)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

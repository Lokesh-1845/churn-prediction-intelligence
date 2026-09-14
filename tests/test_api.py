from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_health():
    print("\n==========================================")
    print("         FASTAPI HEALTH CHECK TEST        ")
    print("==========================================")
    print("Sending GET request to '/'...")
    
    response = client.get("/")
    
    print("\n--- Response Overview ---")
    print(f"Status Code: {response.status_code}")
    print(f"Content Type: {response.headers.get('content-type')}")
    
    print("\n--- Response Payload ---")
    print(response.json())
    
    # Log individual field values
    response_data = response.json()
    print("\n--- Key Diagnostics ---")
    print(f"Health Status: {response_data.get('status')}")
    print(f"Service Name:  {response_data.get('service')}")
    print("==========================================\n")

if __name__ == "__main__":
    test_api_health()
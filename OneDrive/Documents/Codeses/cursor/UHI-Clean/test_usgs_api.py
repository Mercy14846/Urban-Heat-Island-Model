import requests
import json
import time
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
API_URL = "https://m2m.cr.usgs.gov/api/api/json/stable"
API_TOKEN = "25GSdlxIfQdEJiELLqfCgfd34kT8wGyrabBvwJi30r8EiPIThSRVhXsNSM6zS!wm"
USERNAME = "Mercy148464"
CATALOG_ID = "EE"

def test_login():
    """Test login with token to the USGS API"""
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "username": USERNAME,
        "token": API_TOKEN,
        "catalogId": CATALOG_ID,
        "authType": "EROS"
    }
    
    try:
        response = requests.post(
            f"{API_URL}/login-token",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Login response: {data}")
            auth_token = data.get('data')
            if auth_token:
                logger.info("✓ Login successful!")
                return auth_token
            else:
                logger.error("No token in response")
                return None
        else:
            logger.error(f"Login failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        return None

def test_permissions(auth_token):
    """Test getting permissions from the API"""
    headers = {
        "Content-Type": "application/json",
        "X-Auth-Token": auth_token
    }
    
    try:
        response = requests.get(
            f"{API_URL}/permissions",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Permissions: {data}")
            return data.get('data', [])
        else:
            logger.error(f"Permissions request failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return []
    except Exception as e:
        logger.error(f"Error during permissions request: {str(e)}")
        return []

def test_search(auth_token):
    """Test a simple search with minimal parameters"""
    headers = {
        "Content-Type": "application/json",
        "X-Auth-Token": auth_token
    }
    
    # Very simple query with minimal parameters
    payload = {
        "datasetName": "landsat_ot_c2_l1",
        "maxResults": 2,
        "startingNumber": 1,
        "metadataType": "summary",
        "catalogId": CATALOG_ID,
        "sceneFilter": {
            "spatialFilter": None,
            "acquisitionFilter": {
                "start": "2023-01-01",
                "end": "2023-01-02"
            }
        }
    }
    
    try:
        logger.info("Sending search request...")
        response = requests.post(
            f"{API_URL}/scene-search",
            json=payload,
            headers=headers,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Search response: {json.dumps(data, indent=2)}")
            return data
        else:
            logger.error(f"Search failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error during search request: {str(e)}")
        return None

def main():
    """Main test function"""
    auth_token = test_login()
    if auth_token:
        permissions = test_permissions(auth_token)
        if "download" in permissions:
            logger.info("User has download permission!")
            
        search_results = test_search(auth_token)
        if search_results:
            logger.info("Search test successful!")
        else:
            logger.error("Search test failed.")
    else:
        logger.error("Login failed. Cannot proceed with tests.")

if __name__ == "__main__":
    main() 
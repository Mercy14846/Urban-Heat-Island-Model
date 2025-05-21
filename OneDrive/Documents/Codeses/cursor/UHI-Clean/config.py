import os

# Earth Explorer credentials
EARTHEXPLORER_USERNAME = "Mercy148464"
EARTHEXPLORER_PASSWORD = "onaopemipo123"

# M2M API Configuration
M2M_API_URL = "https://m2m.cr.usgs.gov/api/api/json/stable"
API_TOKEN_EXPIRATION_DAYS = 14
API_CATALOG_ID = "EE"

# M2M API Token (generated from https://ers.cr.usgs.gov/profile)
# Make sure this is your actual token from ERS without any Bearer prefix
M2M_API_TOKEN = "25GSdlxIfQdEJiELLqfCgfd34kT8wGyrabBvwJi30r8EiPIThSRVhXsNSM6zS!wm"

# Endpoints
LOGIN_TOKEN_ENDPOINT = "login-token"
LOGIN_ENDPOINT = "login"
SEARCH_ENDPOINT = "scene-search"
DATASET_SEARCH_ENDPOINT = "dataset-search"
METADATA_ENDPOINT = "metadata"
DOWNLOAD_ENDPOINT = "download"
DOWNLOAD_OPTIONS_ENDPOINT = "download-options"
DOWNLOAD_REQUEST_ENDPOINT = "download-request"

# Request timeouts and retry settings
CONNECT_TIMEOUT = 30  # Reduced timeout
READ_TIMEOUT = 60    # Reduced timeout
MAX_RETRIES = 3
RETRY_BACKOFF = 5

# SSL verification
VERIFY_SSL = True  # Enable SSL verification for production

# If environment variables are not set, use these default values
# Replace these with your actual Earth Explorer credentials
if not EARTHEXPLORER_USERNAME or not EARTHEXPLORER_PASSWORD:
    EARTHEXPLORER_USERNAME = "Mercy148464"  # Replace with your username
    EARTHEXPLORER_PASSWORD = "onaopemipo123"  # Replace with your password 
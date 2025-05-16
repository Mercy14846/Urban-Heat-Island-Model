import os

# Earth Explorer credentials
EARTHEXPLORER_USERNAME = os.getenv('EARTHEXPLORER_USERNAME')
EARTHEXPLORER_PASSWORD = os.getenv('EARTHEXPLORER_PASSWORD')

# If environment variables are not set, use these default values
# Replace these with your actual Earth Explorer credentials
if not EARTHEXPLORER_USERNAME or not EARTHEXPLORER_PASSWORD:
    EARTHEXPLORER_USERNAME = "Mercy148464"  # Replace with your username
    EARTHEXPLORER_PASSWORD = "onaopemipo123"  # Replace with your password 
# Urban Heat Island Model QGIS Plugin

This QGIS plugin implements a comprehensive toolset for **detecting**, **forecasting**, and **predicting** Urban Heat Islands (UHI) using Landsat 8 satellite imagery. It features direct integration with the USGS Earth Explorer M2M API for seamless data acquisition.

![QGIS View](https://github.com/user-attachments/assets/07a0c242-517f-47f6-8c8e-c08a706fb773)

## Key Features

### 1. Data Acquisition (M2M API)
- **Direct Search**: Search for Landsat scenes directly within QGIS using your M2M API credentials.
- **Criteria**: Filter by Map Canvas Extent, Date Range, and Cloud Cover.
- **Auto-Download**: Automatically downloads and organizes the required bands (B4, B5, B6, B10) for analysis.

### 2. Analysis Modes
- **Detection (Spectral Indices)**:
    - Uses Red, NIR, SWIR, and Thermal bands.
    - Calculates NDVI, NDBI, and UI to compute a weighted UHI Index.
- **Forecast (Scenario-Based)**:
    - Simulates urban growth scenarios (e.g., +10% urbanization).
    - Adjusts NDBI values to model future heat distribution.
- **Prediction (Machine Learning)**:
    - **Random Forest**: Runs a lightweight regression model to predict thermal anomalies.
    - **Deep Learning (U-Net)**: (Requires TensorFlow) Uses a trained U-Net model for advanced spatial pattern recognition.

### 3. Visualization
- Automatically loads results as new raster layers in QGIS.
- Exports results to GeoTIFF for further processing.

## Installation

### Prerequisites
- **QGIS 3.x**
- **Python 3.8+** (bundled with QGIS or external)
- **USGS Earth Explorer Account** (for data download)

### Python Dependencies
The plugin requires specific Python packages. Open the **OSGeo4W Shell** (Windows) or your terminal and run:

```bash
pip install requests scikit-image scikit-learn numpy
# Optional: For Deep Learning mode
pip install tensorflow
```

### Plugin Installation
1.  Navigate to your QGIS plugins folder:
    *   **Windows**: `C:\Users\<YourUser>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins`
    *   **Mac/Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins`
2.  Clone this repository into the `plugins` folder:
    ```bash
    git clone https://github.com/Mercy14846/Urban-Heat-Island-Model.git uhi_qgis_plugin
    ```
    *(Note: Ensure the folder is named `uhi_qgis_plugin` so QGIS recognizes it.)*
3.  Restart QGIS.
4.  Go to **Plugins > Manage and Install Plugins > Installed** and check "Urban Heat Island Model".

## Setup & Configuration

### USGS Credentials
To use the search and download features, you need a USGS account with M2M API access requested.

1.  **In Plugin**: Enter your Username and Password directly in the "Data Acquisition" tab.
2.  **Config File**: Alternatively, edit `config.py` in the plugin directory to set default credentials:
    ```python
    EARTHEXPLORER_USERNAME = "YourUsername"
    M2M_API_TOKEN = "YourToken" # Optional: Use Token directly
    ```

## Usage Workflow

1.  **Open Plugin**: Click the UHI icon in the QGIS toolbar.
2.  **Search Data**:
    *   Go to **Data Acquisition** tab.
    *   Zoom to your area of interest on the map.
    *   Click **Use Map Canvas Extent**.
    *   Set Date Range and Cloud Cover.
    *   Click **Search Scenes**.
3.  **Download**:
    *   Select a scene from the results table.
    *   Click **Download Data**.
4.  **Run Analysis**:
    *   Go to **Analysis** tab.
    *   **Select Mode**: Detection, Forecast, or Prediction.
    *   **Select Bands**: Map the downloaded files (Red, NIR, SWIR, Thermal).
    *   **Output**: Choose a save location.
    *   Click **Run Analysis**.

## Troubleshooting

-   **"Python not found"**: Ensure your environment variables point to the QGIS Python interpreter.
-   **"Missing Dependencies"**: If `tensorflow` is missing, the Deep Learning mode will be disabled. Install it via OSGeo4W Shell.
-   **Auth Errors**: Verify your USGS password or M2M Token. Tokens expire periodically.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

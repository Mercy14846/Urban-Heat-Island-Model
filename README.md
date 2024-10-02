# Urban Heat Island (UHI) Detection Model
## Project Overview
This project involves building a deep-detection model to identify Urban Heat Islands (UHIs) using satellite imagery (e.g., Landsat 8). The model is built using Python, TensorFlow, and several geospatial libraries such as Rasterio and GDAL. The results are exported as GeoTIFF files for further analysis in QGIS or other GIS platforms.

## Features
1. ### Download Satellite Images: Downloads Landsat images using an API (e.g., USGS Earth Explorer or AWS Landsat).
2. Preprocessing:
- Loads and resamples satellite imagery to a common resolution.
- Applies masks to isolate urban areas for UHI detection.
- Normalizes bands for calculating NDVI and other environmental indicators.
3. U-Net CNN for UHI Detection: Implements a U-Net model to detect UHI areas using satellite data.
Post-processing:
Exports results as GeoTIFF files for use in GIS software.
Integration with QGIS: The results can be visualized in QGIS to aid in urban planning and climate analysis.
Requirements
Python Libraries:
numpy
rasterio
gdal
requests
tensorflow
scikit-learn
Installation:
Install Python (version >= 3.8 recommended).

Install the required packages by running the following command:

bash
Copy code
pip install numpy rasterio gdal requests tensorflow scikit-learn
Install the Google Earth Engine API:

bash
Copy code
pip install earthengine-api
GDAL Installation:
GDAL is required for reading/writing geospatial data. Install GDAL using:

bash
Copy code
conda install -c conda-forge gdal
Or via apt for Linux:

bash
Copy code
sudo apt install gdal-bin
TensorFlow Installation:
TensorFlow can be installed with pip:

bash
Copy code
pip install tensorflow
Data Download
You can download Landsat imagery from:

USGS Earth Explorer: https://earthexplorer.usgs.gov/
Google Earth Engine: https://earthengine.google.com/
Amazon Web Services (AWS) Landsat Public Dataset: https://registry.opendata.aws/landsat-8/
Once downloaded, store the image files in the appropriate directory for loading into the code.

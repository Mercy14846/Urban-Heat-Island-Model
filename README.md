# Urban Heat Island (UHI) Detection Model
## Project Overview
This project involves building a deep-detection model to identify Urban Heat Islands (UHIs) using satellite imagery (e.g., Landsat 8). The model is built using Python, TensorFlow, and several geospatial libraries such as Rasterio and GDAL. The results are exported as GeoTIFF files for further analysis in QGIS or other GIS platforms.

## Features
1. Download Satellite Images: Downloads Landsat images using an API (e.g., USGS Earth Explorer or AWS Landsat).
2. Preprocessing:
- Loads and resamples satellite imagery to a common resolution.
- Applies masks to isolate urban areas for UHI detection.
- Normalizes bands for calculating NDVI and other environmental indicators.
3. U-Net CNN for UHI Detection: Implements a U-Net model to detect UHI areas using satellite data.
4. Post-processing:
- Exports results as GeoTIFF files for use in GIS software.
5. Integration with QGIS: The results can be visualized in QGIS to aid in urban planning and climate analysis.

# Requirements
**Python Libraries:**
- numpy
- rasterio
- gdal
- requests
- tensorflow
- scikit-learn

# Installation:
1. Install Python (version >= 3.8 recommended).

2. Install the required packages by running the following command:
bash
```Copy code
pip install numpy rasterio gdal requests tensorflow scikit-learn
```
3. Install the Google Earth Engine API:

bash
Copy code
pip install earthengine-api

## GDAL Installation:
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


## Data Download
You can download Landsat imagery from:

- USGS Earth Explorer: https://earthexplorer.usgs.gov/
- Google Earth Engine: https://earthengine.google.com/
- Amazon Web Services (AWS) Landsat Public Dataset: https://registry.opendata.aws/landsat-8/

Once downloaded, store the image files in the appropriate directory for loading into the code.

Running the Code
Download Landsat Imagery: Modify the landsat_url variable in the code to point to the actual download location or download the files manually from one of the sources mentioned above.

Example:

python
Copy code
landsat_url = "https://example.com/path/to/landsat_image.tif"
save_path = "landsat_image.tif"
download_landsat_image(landsat_url, save_path)
Loading and Preprocessing: The script resamples the image to a common resolution and applies a mask to isolate urban areas.

U-Net Model: The model is defined using TensorFlow and can be trained on the preprocessed satellite data to detect UHIs. Ensure that you have enough labeled data to train the model.

Exporting to GeoTIFF: Once the model has made predictions, the output is saved as a GeoTIFF, which can be opened in QGIS or any other GIS platform for visualization.

python
Copy code
export_to_geotiff(masked_image, output_path, new_profile)
QGIS Integration:

Open QGIS and load the output GeoTIFF file (uhi_prediction.tif).
Use QGIS tools to further analyze the UHI predictions, visualize results, or combine them with other geospatial layers.
Example Workflow
Download Landsat Image:

python
Copy code
landsat_url = "https://example.com/path/to/landsat_image.tif"
save_path = "landsat_image.tif"
download_landsat_image(landsat_url, save_path)
Preprocess Image:

python
Copy code
image, profile = load_satellite_image(save_path)
resampled_image, new_profile = resample_image(image, profile, 30)  # Example 30m resolution
Calculate NDVI:

python
Copy code
ndvi = calculate_ndvi(nir_band, red_band)
Build and Train U-Net Model:

python
Copy code
model = build_unet_model((128, 128, 3))
train_model(model, train_data, train_labels, val_data, val_labels)
Export to GeoTIFF:

python
Copy code
export_to_geotiff(prediction, "uhi_prediction.tif", new_profile)
Troubleshooting
GDAL Issues: If you face issues with GDAL (e.g., not installed or not found), ensure it's properly installed via Conda or your system's package manager.
Module Not Found: Ensure all dependencies (GDAL, Rasterio, TensorFlow) are installed in the correct Python environment.
Future Enhancements
Integrate additional satellite data sources like Sentinel-2.
Refine the model for better UHI detection accuracy.
Add more case studies in diverse urban environments.
Written by:
Akintola Mercy Ọ̀nàọpẹ́mipọ̀ is a Geospatial Developer with an advance special interest in geospatial community development.
License
This project is open-source and distributed under the MIT License.

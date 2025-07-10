import numpy as np
import tensorflow as tf
from osgeo import gdal
from arcgis.raster import Raster
import rasterio
from rasterio.plot import show
from rasterio.features import shapes
import ee

# Authenticate and initialize the Earth Engine API
ee.Authenticate()  # You'll be prompted to log in with your Google account
ee.Initialize()

# Define the region of interest (ROI) - here, a simple bounding box
roi = ee.Geometry.Rectangle([12.45, 41.9, 12.55, 42.0])  # Example coordinates (Rome, Italy)

# Get Landsat 8 surface reflectance data
landsat = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
           .filterDate('2023-06-01', '2023-06-30') \
           .filterBounds(roi) \
           .map(lambda image: image.clip(roi))

# Select bands for NDVI (Near Infrared - Red) and LST (Thermal Infrared)
ndvi = landsat.map(lambda image: image.normalizedDifference(['B5', 'B4']).rename('NDVI'))
lst = landsat.map(lambda image: image.select('B10').multiply(0.1).rename('LST'))

# Reduce collections to single images (e.g., mean over the time period)
ndvi_mean = ndvi.mean().clip(roi)
lst_mean = lst.mean().clip(roi)

# Export to Google Drive (or download directly if small area)
export_ndvi = ee.batch.Export.image.toDrive(ndvi_mean, description='NDVI_Export', scale=30, region=roi.getInfo())
export_lst = ee.batch.Export.image.toDrive(lst_mean, description='LST_Export', scale=30, region=roi.getInfo())

# Start the export tasks
export_ndvi.start()
export_lst.start()

# Load the LST GeoTIFF
with rasterio.open('LST_data.tif') as src:
    lst_data = src.read(1)
    lst_profile = src.profile

# Load the NDVI GeoTIFF
with rasterio.open('NDVI_data.tif') as src:
    ndvi_data = src.read(1)
    ndvi_profile = src.profile

# Display the LST data
show(lst_data, title="Land Surface Temperature (LST)")

# Display the NDVI data
show(ndvi_data, title="Normalized Difference Vegetation Index (NDVI)")

# # Load and preprocess data
def preprocess_data(input_path):
    ds = gdal.Open(input_path)
    array = ds.ReadAsArray()
    return np.expand_dims(array, axis=-1)

# Define the CNN model
def create_uhi_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training
model = create_uhi_model()
train_data = preprocess_data('train.tif')
train_labels = np.array([1, 0, 1, 0])  # Example labels
model.fit(train_data, train_labels, epochs=10)

# Save output for ArcGIS
output = model.predict(train_data)
gdal_array.SaveArray(output, 'uhi_detection.tif', format="GTiff")

# Load into ArcGIS
uhi_raster = Raster('uhi_detection.tif')
uhi_raster_layer = uhi_raster.layers[0]
uhi_raster_layer.save('UHI_Visualization')

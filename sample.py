import ee

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize()

# Define a larger region of interest (ROI)
roi = ee.Geometry.Polygon([
    [[-74.10, 40.55], [-74.10, 40.85], [-73.70, 40.85], [-73.70, 40.55]]  # Example coordinates (New York City)
])

# Get Landsat 8 surface reflectance data for a large region
landsat = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
           .filterDate('2023-06-01', '2023-06-30') \
           .filterBounds(roi)

# NDVI and LST calculation for large areas
ndvi = landsat.map(lambda image: image.normalizedDifference(['B5', 'B4']).rename('NDVI'))
lst = landsat.map(lambda image: image.select('B10').multiply(0.1).rename('LST'))

# Mean NDVI and LST over the time period
ndvi_mean = ndvi.mean().clip(roi)
lst_mean = lst.mean().clip(roi)

# Export the results to Google Drive (alternatively use Google Cloud)
export_ndvi = ee.batch.Export.image.toDrive(ndvi_mean, description='NDVI_Large_Scale_Export', scale=30, region=roi.getInfo())
export_lst = ee.batch.Export.image.toDrive(lst_mean, description='LST_Large_Scale_Export', scale=30, region=roi.getInfo())

# Start the export tasks
export_ndvi.start()
export_lst.start()

import rasterio
import numpy as np

# Load large-scale GeoTIFF (e.g., exported LST data)
def load_large_raster(input_path):
    with rasterio.open(input_path) as dataset:
        # Read the raster data efficiently
        for i in range(1, dataset.count + 1):
            raster_chunk = dataset.read(i, window=((0, 1024), (0, 1024)))  # Example chunk size
            yield raster_chunk

# Example: Load a chunk of the raster data
for raster_chunk in load_large_raster('LST_large.tif'):
    print(raster_chunk.shape)  # Process in chunks to avoid memory overload


import tensorflow as tf

# Define the CNN model
def create_uhi_model(input_shape=(256, 256, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training data can now be processed in chunks
model = create_uhi_model(input_shape=(1024, 1024, 1))

# Assume that `train_chunk` is a chunk of training data
for train_chunk in load_large_raster('train_large.tif'):
    train_chunk = np.expand_dims(train_chunk, axis=-1)  # Expand dims for model input
    train_labels = np.array([1] * train_chunk.shape[0])  # Placeholder labels
    model.fit(train_chunk, train_labels, epochs=2)


import gdal
from rasterio.transform import from_origin
import numpy as np

# Function to save model output to GeoTIFF format
def save_model_output_as_geotiff(output, output_path, profile):
    output = np.squeeze(output, axis=-1)  # Ensure the output shape is correct
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(output, 1)

# Example profile (you can load it from an actual GeoTIFF file)
profile = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'width': 1024,
    'height': 1024,
    'count': 1,
    'crs': 'EPSG:4326',
    'transform': from_origin(-74.10, 40.85, 0.0001, 0.0001)  # Example geo-transform
}

# Save output of UHI detection model
uhi_output = model.predict(train_chunk)
save_model_output_as_geotiff(uhi_output, 'uhi_output_large.tif', profile)

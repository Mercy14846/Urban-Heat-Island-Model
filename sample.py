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



# Importing necessary libraries for geospatial processing and machine learning
import rasterio
import numpy as np
import requests
from osgeo import gdal
from rasterio.enums import Resampling
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# Example: Downloading Landsat 8 imagery using requests
def download_landsat_image(url, save_path):
    """Downloads Landsat 8 satellite image."""
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

# Load satellite image (example using Rasterio)
def load_satellite_image(file_path):
    """Loads satellite image using Rasterio."""
    dataset = rasterio.open(file_path)
    return dataset.read(1), dataset.profile

# Example URLs and file paths
landsat_url = "https://landsat-pds.s3.amazonaws.com/c1/L8/001/002/LC08_L1TP_001002_20210320_20210330_01_T1/LC08_L1TP_001002_20210320_20210330_01_T1_B4.TIF"  # Adjust URL as needed
save_path = "B4.TIF"

# Downloading and loading the image
download_landsat_image(landsat_url, save_path)
image, profile = load_satellite_image(save_path)

# Resample the image to a common resolution (e.g., 30m resolution)
def resample_image(image, profile, target_resolution):
    """Resamples the image to a common resolution for analysis."""
    new_width = int(profile['width'] * profile['transform'][0] / target_resolution)
    new_height = int(profile['height'] * profile['transform'][0] / target_resolution)
    data = image.read(
        out_shape=(image.count, new_height, new_width),
        resampling=Resampling.bilinear
    )
    profile.update({
        'width': new_width,
        'height': new_height,
        'transform': image.transform * image.transform.scale(
            (image.width / new_width),
            (image.height / new_height)
        )
    })
    return data, profile

# Example: Masking non-urban areas
def apply_mask(image, mask):
    """Applies a mask to the satellite image to isolate urban areas."""
    return np.where(mask, image, np.nan)

# Resample the image and apply a mask
target_resolution = 30  # Example resolution in meters
resampled_image, new_profile = resample_image(image, profile, target_resolution)
mask = np.ones_like(resampled_image, dtype=bool)  # Example mask, this would need to be generated based on urban area data
masked_image = apply_mask(resampled_image, mask)

# Normalization of features (e.g., NDVI)
def normalize_data(data):
    """Normalizes data for machine learning input."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)

# Calculate NDVI from NIR and Red bands
def calculate_ndvi(nir_band, red_band):
    """Calculates Normalized Difference Vegetation Index (NDVI)."""
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    return normalize_data(ndvi)

# Load NIR and Red bands and calculate NDVI
nir_band, _ = load_satellite_image("nir_band.tif")
red_band, _ = load_satellite_image("red_band.tif")
ndvi = calculate_ndvi(nir_band, red_band)

# Deep Learning Model: U-Net for UHI detection
def build_unet_model(input_shape):
    """Builds a U-Net model for Urban Heat Island detection."""
    inputs = layers.Input(shape=input_shape)

    # Encoder (Downsampling)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.MaxPooling2D((2, 2))(c1)
    # Additional layers for the encoder would be added here

    # Decoder (Upsampling)
    c9 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c1)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Example: Compiling and training the U-Net model
def train_model(model, train_data, train_labels, val_data, val_labels, epochs=10, batch_size=16):
    """Compiles and trains the U-Net model."""
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=epochs, batch_size=batch_size)

# Model building example usage
input_shape = (128, 128, 3)  # Example input shape based on the resolution of the resampled image
model = build_unet_model(input_shape)

# You would need to prepare training data (train_data, train_labels) and validation data (val_data, val_labels)
train_model(model, train_data, train_labels, val_data, val_labels)

# Function to export model prediction to GeoTIFF format
def export_to_geotiff(prediction, output_path, profile):
    """Exports the prediction result to a GeoTIFF file."""
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prediction, 1)

# Example: Exporting the model’s output
output_path = "uhi_prediction.tif"
export_to_geotiff(masked_image, output_path, new_profile)

# Function to evaluate the model's performance
def evaluate_model(model, test_data, test_labels):
    """Evaluates the model's performance using test data."""
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Model evaluation example usage
accuracy = evaluate_model(model, test_data, test_labels)

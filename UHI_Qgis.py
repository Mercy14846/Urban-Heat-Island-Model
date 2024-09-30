import rasterio
import numpy as np
import requests
from osgeo import gdal

# Example: Downloading Landsat 8 imagery using requests (you'll need to adapt this to the specific API you are using)
def download_landsat_image(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

# Load satellite image (example using Rasterio)
def load_satellite_image(file_path):
    dataset = rasterio.open(file_path)
    return dataset.read(1), dataset.profile

# Example URLs and file paths
landsat_url = "https://example.com/path/to/landsat_image.tif"
save_path = "landsat_image.tif"

# Downloading and loading the image
download_landsat_image(landsat_url, save_path)
image, profile = load_satellite_image(save_path)

from rasterio.enums import Resampling

# Resample to a common resolution
def resample_image(image, profile, target_resolution):
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

# Masking non-urban areas
def apply_mask(image, mask):
    return np.where(mask, image, np.nan)

# Example usage
target_resolution = 30  # Example resolution in meters
resampled_image, new_profile = resample_image(image, profile, target_resolution)

# Example mask (you need to create a proper mask based on your dataset)
mask = np.ones_like(resampled_image, dtype=bool)
masked_image = apply_mask(resampled_image, mask)


from sklearn.preprocessing import MinMaxScaler

# Normalization of features
def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)

# Calculate NDVI from NIR and Red bands
def calculate_ndvi(nir_band, red_band):
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    return normalize_data(ndvi)

# Example: Calculate NDVI
nir_band, _ = load_satellite_image("nir_band.tif")
red_band, _ = load_satellite_image("red_band.tif")

ndvi = calculate_ndvi(nir_band, red_band)


import tensorflow as tf
from tensorflow.keras import layers, models

# Example: U-Net model architecture
def build_unet_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.MaxPooling2D((2, 2))(c1)
    # (Additional layers would be added here)

    # Decoder
    c9 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c1)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Compile and train the model
def train_model(model, train_data, train_labels, val_data, val_labels, epochs=10, batch_size=16):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=epochs, batch_size=batch_size)

# Example usage
input_shape = (128, 128, 3)  # Example input shape
model = build_unet_model(input_shape)
# You would need to prepare train_data, train_labels, val_data, and val_labels
# train_model(model, train_data, train_labels, val_data, val_labels)

import rasterio
from rasterio.transform import from_origin

# Export model output to GeoTIFF
def export_to_geotiff(prediction, output_path, profile):
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prediction, 1)

# Example usage
output_path = "uhi_prediction.tif"
export_to_geotiff(masked_image, output_path, new_profile)


# Function to evaluate the model performance
def evaluate_model(model, test_data, test_labels):
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Example testing and refinement
# accuracy = evaluate_model(model, test_data, test_labels)
# if accuracy < desired_threshold:
#     # Adjust hyperparameters, retrain, or refine features

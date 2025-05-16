# Importing necessary libraries for geospatial processing and machine learning
import os
import logging
import rasterio
import numpy as np
import requests
from osgeo import gdal
from rasterio.enums import Resampling
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime
import json
from usgs import api
from config import EARTHEXPLORER_USERNAME, EARTHEXPLORER_PASSWORD

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# USGS Earth Explorer constants
LANDSAT_COLLECTION = "LANDSAT_8_C1"

class UHIModel:
    def __init__(self, data_dir="data", ee_username=None, ee_password=None):
        """Initialize the Urban Heat Island Model."""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.model = None
        
        # Earth Explorer credentials
        self.ee_username = ee_username or EARTHEXPLORER_USERNAME
        self.ee_password = ee_password or EARTHEXPLORER_PASSWORD
        
        if not self.ee_username or not self.ee_password:
            raise ValueError("Earth Explorer credentials are required. Please set them in config.py or environment variables.")
        
        if self.ee_username == "your_username" or self.ee_password == "your_password":
            raise ValueError("Please replace the default credentials in config.py with your actual Earth Explorer credentials.")
        
        # Initialize USGS API session
        try:
            # Login to USGS
            api_key = api.login(self.ee_username, self.ee_password)
            if not api_key:
                raise ValueError("Failed to authenticate with USGS Earth Explorer")
            
            logger.info("Successfully authenticated with USGS Earth Explorer")
            
        except Exception as e:
            logger.error(f"Failed to authenticate with USGS Earth Explorer: {str(e)}")
            raise

    def download_landsat_image(self, scene_id, band_number, save_path):
        """Downloads Landsat 8 satellite image using USGS Earth Explorer."""
        try:
            # Get scene metadata
            metadata = api.metadata(LANDSAT_COLLECTION, [scene_id])
            if not metadata or 'data' not in metadata or not metadata['data']:
                raise ValueError(f"Scene {scene_id} not found")

            # Request download
            download_info = api.download(LANDSAT_COLLECTION, [scene_id])
            if not download_info or 'data' not in download_info or not download_info['data']:
                raise ValueError(f"Unable to get download URL for scene {scene_id}")

            # Find the correct band URL
            download_url = None
            for item in download_info['data']:
                if isinstance(item, dict) and 'url' in item:
                    if f"_B{band_number}." in item['url']:
                        download_url = item['url']
                        break

            if not download_url:
                raise ValueError(f"Band {band_number} not found for scene {scene_id}")

            # Download the file
            full_path = os.path.join(self.data_dir, save_path)
            response = requests.get(download_url, stream=True)
            response.raise_for_status()

            with open(full_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Successfully downloaded band {band_number} to {full_path}")
            return full_path

        except Exception as e:
            logger.error(f"Failed to download image: {str(e)}")
            raise

    def load_satellite_image(self, file_path):
        """Loads satellite image using Rasterio with error handling."""
        try:
            dataset = rasterio.open(file_path)
            return dataset.read(1), dataset.profile
        except rasterio.errors.RasterioError as e:
            logger.error(f"Failed to load satellite image: {str(e)}")
            raise

    def resample_image(self, image, profile, target_resolution):
        """Resamples the image to a common resolution for analysis."""
        try:
            with rasterio.open(image) as src:
                new_width = int(src.width * src.transform[0] / target_resolution)
                new_height = int(src.height * src.transform[0] / target_resolution)
                
                data = src.read(
                    out_shape=(src.count, new_height, new_width),
                    resampling=Resampling.bilinear
                )
                
                transform = src.transform * src.transform.scale(
                    (src.width / new_width),
                    (src.height / new_height)
                )
                
                profile.update({
                    'width': new_width,
                    'height': new_height,
                    'transform': transform
                })
                
                return data, profile
        except Exception as e:
            logger.error(f"Failed to resample image: {str(e)}")
            raise

    def calculate_ndvi(self, nir_path, red_path):
        """Calculates NDVI from NIR and Red bands with error handling."""
        try:
            nir_band, _ = self.load_satellite_image(nir_path)
            red_band, _ = self.load_satellite_image(red_path)
            
            # Avoid division by zero
            denominator = (nir_band + red_band)
            denominator[denominator == 0] = 1e-10
            
            ndvi = (nir_band - red_band) / denominator
            return self.normalize_data(ndvi)
        except Exception as e:
            logger.error(f"Failed to calculate NDVI: {str(e)}")
            raise

    def normalize_data(self, data):
        """Normalizes data for machine learning input."""
        scaler = MinMaxScaler()
        return scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)

    def build_unet_model(self, input_shape):
        """Builds a complete U-Net model for Urban Heat Island detection."""
        inputs = layers.Input(shape=input_shape)

        # Encoder
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        # Bridge
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)

        # Decoder
        up1 = layers.UpSampling2D(size=(2, 2))(conv3)
        up1 = layers.Conv2D(128, 2, activation='relu', padding='same')(up1)
        concat1 = layers.concatenate([conv2, up1], axis=3)
        conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat1)

        up2 = layers.UpSampling2D(size=(2, 2))(conv4)
        up2 = layers.Conv2D(64, 2, activation='relu', padding='same')(up2)
        concat2 = layers.concatenate([conv1, up2], axis=3)
        conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat2)

        outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)

        model = models.Model(inputs=[inputs], outputs=[outputs])
        return model

    def train_model(self, train_data, train_labels, val_data=None, val_labels=None, 
                   epochs=10, batch_size=16):
        """Trains the U-Net model with error handling."""
        try:
            if self.model is None:
                input_shape = train_data.shape[1:]
                self.model = self.build_unet_model(input_shape)
            
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(self.data_dir, 'best_model.h5'),
                    save_best_only=True
                )
            ]
            
            history = self.model.fit(
                train_data, train_labels,
                validation_data=(val_data, val_labels) if val_data is not None else None,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
            
            return history
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise

    def predict(self, input_data):
        """Makes predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(input_data)

    def export_prediction(self, prediction, output_path, profile):
        """Exports the prediction result to a GeoTIFF file."""
        try:
            full_path = os.path.join(self.data_dir, output_path)
            with rasterio.open(full_path, 'w', **profile) as dst:
                dst.write(prediction, 1)
            logger.info(f"Successfully exported prediction to {full_path}")
        except Exception as e:
            logger.error(f"Failed to export prediction: {str(e)}")
            raise

    def __del__(self):
        """Cleanup when the object is destroyed."""
        try:
            api.logout()
            logger.info("Successfully logged out from USGS Earth Explorer")
        except:
            pass

def main():
    """Main execution function."""
    try:
        # Initialize the model with Earth Explorer credentials
        uhi_model = UHIModel()
        
        # Example Landsat 8 scene ID (using a more recent scene)
        scene_id = "LC08_L1GT_044034_20230415_20230416_02_T2"
        
        # Download and process images
        logger.info("Downloading red band...")
        red_path = uhi_model.download_landsat_image(scene_id, "4", 'red_band.tif')
        
        logger.info("Downloading NIR band...")
        nir_path = uhi_model.download_landsat_image(scene_id, "5", 'nir_band.tif')
        
        # Calculate NDVI
        logger.info("Calculating NDVI...")
        ndvi = uhi_model.calculate_ndvi(nir_path, red_path)
        
        # Here you would normally prepare your training data
        logger.info("Preparing training data...")
        input_shape = (128, 128, 1)
        dummy_train_data = np.random.random((100, *input_shape))
        dummy_train_labels = np.random.randint(0, 2, (100, 128, 128, 1))
        
        # Train the model
        logger.info("Training model...")
        history = uhi_model.train_model(
            dummy_train_data,
            dummy_train_labels,
            epochs=5
        )
        
        logger.info("Training completed successfully")
        
    except ValueError as ve:
        logger.error(f"Configuration error: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"An error occurred in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

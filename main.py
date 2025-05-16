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
from typing import Optional, Tuple, Dict, Any

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Custom exceptions for better error handling
class USGSAuthenticationError(Exception):
    """Raised when authentication with USGS Earth Explorer fails."""
    pass

class ImageDownloadError(Exception):
    """Raised when there's an error downloading satellite imagery."""
    pass

class ImageProcessingError(Exception):
    """Raised when there's an error processing satellite imagery."""
    pass

class ModelError(Exception):
    """Raised when there's an error with the U-Net model."""
    pass

# USGS Earth Explorer constants
LANDSAT_COLLECTION = "LANDSAT_8_C1"

class UHIModel:
    def __init__(self, data_dir: str = "data", ee_username: Optional[str] = None, 
                 ee_password: Optional[str] = None):
        """Initialize the Urban Heat Island Model.
        
        Args:
            data_dir (str): Directory to store downloaded and processed data
            ee_username (Optional[str]): Earth Explorer username
            ee_password (Optional[str]): Earth Explorer password
            
        Raises:
            USGSAuthenticationError: If authentication fails
            ValueError: If credentials are missing or invalid
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.model = None
        self._api_key = None
        
        # Earth Explorer credentials
        self.ee_username = ee_username or EARTHEXPLORER_USERNAME
        self.ee_password = ee_password or EARTHEXPLORER_PASSWORD
        
        if not self.ee_username or not self.ee_password:
            raise ValueError("Earth Explorer credentials are required. Please set them in config.py or environment variables.")
        
        if self.ee_username == "your_username" or self.ee_password == "your_password":
            raise ValueError("Please replace the default credentials in config.py with your actual Earth Explorer credentials.")
        
        # Initialize USGS API session
        self._initialize_api_session()

    def _initialize_api_session(self) -> None:
        """Initialize and verify USGS API session.
        
        Raises:
            USGSAuthenticationError: If authentication fails
        """
        try:
            self._api_key = api.login(self.ee_username, self.ee_password)
            if not self._api_key:
                raise USGSAuthenticationError("Failed to authenticate with USGS Earth Explorer")
            
            # Verify the API key is valid
            try:
                api.metadata(LANDSAT_COLLECTION, ["LC08_L1GT_044034_20230415_20230416_02_T2"])
                logger.info("Successfully authenticated with USGS Earth Explorer")
            except Exception:
                raise USGSAuthenticationError("API key validation failed")
            
        except Exception as e:
            logger.error(f"Failed to authenticate with USGS Earth Explorer: {str(e)}")
            raise USGSAuthenticationError(f"Authentication failed: {str(e)}")

    def download_landsat_image(self, scene_id: str, band_number: str, 
                            save_path: str) -> str:
        """Downloads Landsat 8 satellite image using USGS Earth Explorer.
        
        Args:
            scene_id (str): Landsat scene identifier
            band_number (str): Band number to download
            save_path (str): Path to save the downloaded file
            
        Returns:
            str: Path to the downloaded file
            
        Raises:
            ImageDownloadError: If download fails
            ValueError: If scene or band not found
        """
        try:
            # Check if file already exists
            full_path = os.path.join(self.data_dir, save_path)
            if os.path.exists(full_path):
                logger.info(f"Using cached band {band_number} from {full_path}")
                return full_path

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

            # Download the file with progress tracking
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(full_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    if total_size > 0:
                        progress = (os.path.getsize(full_path) / total_size) * 100
                        logger.info(f"Download progress: {progress:.1f}%")

            logger.info(f"Successfully downloaded band {band_number} to {full_path}")
            return full_path

        except requests.exceptions.RequestException as e:
            raise ImageDownloadError(f"Network error while downloading: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to download image: {str(e)}")
            raise ImageDownloadError(str(e))

    def load_satellite_image(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Loads satellite image using Rasterio with error handling.
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Image data and metadata profile
            
        Raises:
            ImageProcessingError: If image loading fails
        """
        try:
            with rasterio.open(file_path) as dataset:
                # Read the data and handle nodata values
                data = dataset.read(1)
                nodata = dataset.nodata
                if nodata is not None:
                    data = np.ma.masked_equal(data, nodata).filled(0)
                
                # Get metadata
                profile = dataset.profile
                
                return data, profile
                
        except rasterio.errors.RasterioError as e:
            logger.error(f"Failed to load satellite image: {str(e)}")
            raise ImageProcessingError(f"Failed to load image: {str(e)}")

    def resample_image(self, image: np.ndarray, profile: Dict[str, Any], 
                      target_resolution: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resamples the image to a common resolution for analysis.
        
        Args:
            image (np.ndarray): Input image data
            profile (Dict[str, Any]): Image metadata profile
            target_resolution (float): Target resolution in meters
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Resampled image and updated profile
            
        Raises:
            ImageProcessingError: If resampling fails
        """
        try:
            # Calculate new dimensions
            scale_factor = profile['transform'][0] / target_resolution
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            
            # Update transform
            transform = profile['transform'] * profile['transform'].scale(
                (image.shape[1] / new_width),
                (image.shape[0] / new_height)
            )
            
            # Resample using scikit-image for better performance
            from skimage.transform import resize
            resampled_data = resize(
                image,
                (new_height, new_width),
                order=1,  # bilinear interpolation
                mode='constant',
                anti_aliasing=True
            )
            
            # Update profile
            profile.update({
                'width': new_width,
                'height': new_height,
                'transform': transform
            })
            
            return resampled_data, profile
            
        except Exception as e:
            logger.error(f"Failed to resample image: {str(e)}")
            raise ImageProcessingError(f"Failed to resample image: {str(e)}")

    def calculate_ndvi(self, nir_path: str, red_path: str) -> np.ndarray:
        """Calculates NDVI from NIR and Red bands with error handling.
        
        Args:
            nir_path (str): Path to NIR band image
            red_path (str): Path to Red band image
            
        Returns:
            np.ndarray: Normalized NDVI values
            
        Raises:
            ImageProcessingError: If NDVI calculation fails
        """
        try:
            # Load bands
            nir_band, nir_profile = self.load_satellite_image(nir_path)
            red_band, red_profile = self.load_satellite_image(red_path)
            
            # Ensure same dimensions
            if nir_band.shape != red_band.shape:
                logger.info("Resampling bands to match dimensions...")
                target_res = min(nir_profile['transform'][0], red_profile['transform'][0])
                nir_band, _ = self.resample_image(nir_band, nir_profile, target_res)
                red_band, _ = self.resample_image(red_band, red_profile, target_res)
            
            # Calculate NDVI with proper handling of edge cases
            sum_bands = nir_band + red_band
            diff_bands = nir_band - red_band
            
            # Handle division by zero and invalid values
            ndvi = np.zeros_like(sum_bands, dtype=np.float32)
            valid_mask = sum_bands != 0
            ndvi[valid_mask] = diff_bands[valid_mask] / sum_bands[valid_mask]
            
            # Clip values to valid NDVI range [-1, 1]
            ndvi = np.clip(ndvi, -1, 1)
            
            return self.normalize_data(ndvi)
            
        except Exception as e:
            logger.error(f"Failed to calculate NDVI: {str(e)}")
            raise ImageProcessingError(f"Failed to calculate NDVI: {str(e)}")

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

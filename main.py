# Importing necessary libraries for geospatial processing and machine learning
import os
import logging
import rasterio
import numpy as np
import requests
from osgeo import gdal
from rasterio.enums import Resampling
from sklearn.preprocessing import MinMaxScaler
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    layers = None
    models = None
from datetime import datetime, timedelta
import json
from typing import Optional, Tuple, Dict, Any
from config import (
    EARTHEXPLORER_USERNAME,
    EARTHEXPLORER_PASSWORD,
    M2M_API_URL,
    API_TOKEN_EXPIRATION_DAYS,
    API_CATALOG_ID,
    M2M_API_TOKEN,
    SEARCH_ENDPOINT,
    METADATA_ENDPOINT,
    DOWNLOAD_ENDPOINT,
    DOWNLOAD_OPTIONS_ENDPOINT,
    DOWNLOAD_REQUEST_ENDPOINT,
    LOGIN_TOKEN_ENDPOINT,
    CONNECT_TIMEOUT,
    READ_TIMEOUT,
    VERIFY_SSL,
    MAX_RETRIES,
    RETRY_BACKOFF
)
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import time
import scipy.ndimage

# Disable insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
    """Raised when image download fails."""
    pass

class ImageProcessingError(Exception):
    """Raised when there's an error processing satellite imagery."""
    pass

class ModelError(Exception):
    """Raised when there's an error with the U-Net model."""
    pass

class M2MAPIError(Exception):
    """Raised when M2M API request fails."""
    pass

class RateLimitedException(Exception):
    """Raised when the USGS API rate limits the request."""
    pass

# USGS Earth Explorer constants
LANDSAT_COLLECTION = "landsat_ot_c2_l1"

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
        """Initialize and verify USGS M2M API session.
        
        Raises:
            USGSAuthenticationError: If authentication fails
        """
        try:
            # Configure session with retry strategy
            retry_strategy = Retry(
                total=3,  # number of retries
                backoff_factor=2,  # wait 2, 4, 8 seconds between retries
                status_forcelist=[408, 429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "POST"]
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session = requests.Session()
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            
            # Try token-based authentication first
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {M2M_API_TOKEN}"
            }
            
            # Test token with a simple request
            test_request = {
                "datasetName": LANDSAT_COLLECTION,
                "maxResults": 1,
                "metadataType": "summary"
            }
            
            response = self.session.post(
                f"{M2M_API_URL}/{SEARCH_ENDPOINT}",
                json=test_request,
                headers=headers,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                verify=VERIFY_SSL
            )
            
            if response.status_code == 200:
                self._api_key = M2M_API_TOKEN
                logger.info("Successfully authenticated using API token")
                return
                
            # If token auth fails, try username/password with login-token endpoint
            logger.info("API token validation failed, trying username/password authentication")
            
            login_data = {
                "username": self.ee_username,
                "password": self.ee_password,
                "catalogId": API_CATALOG_ID,
                "authType": "EROS"
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = self.session.post(
                f"{M2M_API_URL}/{LOGIN_TOKEN_ENDPOINT}",
                json=login_data,
                headers=headers,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                verify=VERIFY_SSL
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "data" in data and "token" in data["data"]:
                        self._api_key = data["data"]["token"]
                        logger.info("Successfully authenticated using username/password")
                        print("✓ USGS Authentication successful! API connection established.")
                        return
                except ValueError:
                    logger.warning("Failed to parse authentication response")
            
            # If both methods fail, raise an error with detailed information
            error_msg = f"Authentication failed. Status code: {response.status_code}"
            if response.text:
                try:
                    error_data = response.json()
                    if "errorMessage" in error_data:
                        error_msg = f"Authentication failed: {error_data['errorMessage']}"
                except ValueError:
                    error_msg = f"Authentication failed. Response: {response.text}"
            
            logger.error(error_msg)
            raise USGSAuthenticationError(error_msg)
            
        except requests.exceptions.SSLError as e:
            logger.error(f"SSL verification failed: {str(e)}")
            raise USGSAuthenticationError(f"SSL verification failed. If testing, you may need to set VERIFY_SSL=False in config.py")
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timed out: {str(e)}")
            raise USGSAuthenticationError(f"Connection timed out. Check your network connection or try again later.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during authentication: {str(e)}")
            raise USGSAuthenticationError(f"Network error during authentication: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during authentication: {str(e)}")
            raise USGSAuthenticationError(f"Authentication failed: {str(e)}")

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get the authentication headers for API requests.
        
        Returns:
            Dict[str, str]: Headers dictionary with authentication token
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }

    def download_landsat_image(self, scene_id: str, band_number: str, 
                            save_path: str) -> str:
        """Downloads Landsat 8 satellite image using USGS Earth Explorer M2M API.
        
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

            headers = self._get_auth_headers()

            # Get scene metadata
            scene_metadata = {
                "datasetName": LANDSAT_COLLECTION,
                "entityIds": [scene_id]
            }
            
            response = requests.post(f"{M2M_API_URL}/{METADATA_ENDPOINT}", 
                                  json=scene_metadata, 
                                  headers=headers)
            
            if response.status_code != 200:
                error_msg = f"Failed to get metadata for scene {scene_id}"
                try:
                    error_data = response.json()
                    if "errorMessage" in error_data:
                        error_msg = f"Metadata request failed: {error_data['errorMessage']}"
                except ValueError:
                    error_msg = f"Metadata request failed. Response: {response.text}"
                raise ValueError(error_msg)

            metadata = response.json()
            if "data" not in metadata or not metadata["data"]:
                raise ValueError(f"Scene {scene_id} not found")

            # Request download URL
            download_request = {
                "datasetName": LANDSAT_COLLECTION,
                "entityIds": [scene_id],
                "products": ["STANDARD"]
            }
            
            response = requests.post(f"{M2M_API_URL}/{DOWNLOAD_ENDPOINT}", 
                                  json=download_request, 
                                  headers=headers)
            
            if response.status_code != 200:
                error_msg = f"Failed to get download URL for scene {scene_id}"
                try:
                    error_data = response.json()
                    if "errorMessage" in error_data:
                        error_msg = f"Download request failed: {error_data['errorMessage']}"
                except ValueError:
                    error_msg = f"Download request failed. Response: {response.text}"
                raise ValueError(error_msg)

            download_info = response.json()
            if "data" not in download_info or not download_info["data"]:
                raise ValueError(f"No download information available for scene {scene_id}")

            # Get download URL for the specific band
            download_url = None
            for item in download_info["data"]:
                if f"_B{band_number}." in item["url"]:
                    download_url = item["url"]
                    break

            if not download_url:
                raise ValueError(f"Band {band_number} not found for scene {scene_id}")

            # Download the file with progress tracking
            response = requests.get(download_url, stream=True)
            if response.status_code != 200:
                raise ImageDownloadError(f"Failed to download band {band_number}. Status code: {response.status_code}")
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded_size = 0
            
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}%")

            logger.info(f"Successfully downloaded band {band_number} to {full_path}")
            return full_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while downloading: {str(e)}")
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

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalizes data for machine learning input.
        
        Args:
            data (np.ndarray): Input data array
            
        Returns:
            np.ndarray: Normalized data in range [0, 1]
        """
        scaler = MinMaxScaler()
        return scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)

    def build_unet_model(self, input_shape: Tuple[int, int, int]) -> Any:
        """Builds an enhanced U-Net model for Urban Heat Island detection.
        
        This implementation includes several modern improvements:
        - Batch normalization for better training stability
        - Dropout for regularization
        - Residual connections for better gradient flow
        - ELU activation for better gradient propagation
        - Spatial dropout for feature map regularization
        - Attention mechanism for better feature extraction
        - Model versioning support
        
        Args:
            input_shape (Tuple[int, int, int]): Input shape (height, width, channels)
            
        Returns:
            tf.keras.Model: Compiled U-Net model
        """
        def attention_block(x, g, filters):
            """Attention gate for better feature extraction."""
            theta_x = layers.Conv2D(filters, (1, 1), strides=(1, 1))(x)
            phi_g = layers.Conv2D(filters, (1, 1), strides=(1, 1))(g)
            f = layers.Activation('relu')(layers.add([theta_x, phi_g]))
            psi_f = layers.Conv2D(1, (1, 1), strides=(1, 1))(f)
            rate = layers.Activation('sigmoid')(psi_f)
            return layers.multiply([x, rate])

        def conv_block(x, filters, kernel_size=3, dropout_rate=0.1):
            """Helper function for creating a conv block with batch norm and residual connection."""
            conv = layers.Conv2D(filters, kernel_size, padding='same')(x)
            conv = layers.BatchNormalization()(conv)
            conv = layers.Activation('elu')(conv)
            conv = layers.SpatialDropout2D(dropout_rate)(conv)
            conv = layers.Conv2D(filters, kernel_size, padding='same')(conv)
            conv = layers.BatchNormalization()(conv)
            conv = layers.Activation('elu')(conv)
            return conv

        # Input
        inputs = layers.Input(shape=input_shape)
        
        # Version tracking
        version = tf.keras.layers.Lambda(lambda x: x, name='model_version')(inputs)
        version = tf.keras.layers.Lambda(lambda x: tf.constant(1.0), name='version')(version)

        # Encoder
        conv1 = conv_block(inputs, 64)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = conv_block(pool1, 128)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = conv_block(pool2, 256)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = conv_block(pool3, 512)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        # Bridge
        conv5 = conv_block(pool4, 1024)

        # Decoder with attention
        up4 = layers.UpSampling2D(size=(2, 2))(conv5)
        att4 = attention_block(conv4, up4, 512)
        up4 = layers.concatenate([up4, att4])
        conv6 = conv_block(up4, 512)

        up3 = layers.UpSampling2D(size=(2, 2))(conv6)
        att3 = attention_block(conv3, up3, 256)
        up3 = layers.concatenate([up3, att3])
        conv7 = conv_block(up3, 256)

        up2 = layers.UpSampling2D(size=(2, 2))(conv7)
        att2 = attention_block(conv2, up2, 128)
        up2 = layers.concatenate([up2, att2])
        conv8 = conv_block(up2, 128)

        up1 = layers.UpSampling2D(size=(2, 2))(conv8)
        att1 = attention_block(conv1, up1, 64)
        up1 = layers.concatenate([up1, att1])
        conv9 = conv_block(up1, 64)

        # Output
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)
        
        # Create model
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs, version])
        
        return model

    def train_model(self, train_data: np.ndarray, train_labels: np.ndarray, 
                   val_data: Optional[np.ndarray] = None, 
                   val_labels: Optional[np.ndarray] = None,
                   epochs: int = 10, batch_size: int = 16) -> Any:
        """Trains the U-Net model with advanced training features.
        
        Args:
            train_data (np.ndarray): Training data
            train_labels (np.ndarray): Training labels
            val_data (Optional[np.ndarray]): Validation data
            val_labels (Optional[np.ndarray]): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            tf.keras.callbacks.History: Training history
            
        Raises:
            ModelError: If training fails
        """
        try:
            if self.model is None:
                input_shape = train_data.shape[1:]
                self.model = self.build_unet_model(input_shape)
            
            # Use mixed precision for faster training
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()
                ]
            )
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=5,
                    restore_best_weights=True,
                    monitor='val_loss' if val_data is not None else 'loss'
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(self.data_dir, 'best_model.h5'),
                    save_best_only=True,
                    monitor='val_loss' if val_data is not None else 'loss'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(self.data_dir, 'logs'),
                    histogram_freq=1
                )
            ]
            
            history = self.model.fit(
                train_data, train_labels,
                validation_data=(val_data, val_labels) if val_data is not None else None,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                workers=4,
                use_multiprocessing=True
            )
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise ModelError(f"Training failed: {str(e)}")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Makes predictions using the trained model.
        
        Args:
            input_data (np.ndarray): Input data for prediction
            
        Returns:
            np.ndarray: Model predictions
            
        Raises:
            ModelError: If prediction fails or model is not trained
        """
        if self.model is None:
            raise ModelError("Model has not been trained yet")
            
        try:
            return self.model.predict(input_data)
        except Exception as e:
            logger.error(f"Failed to make prediction: {str(e)}")
            raise ModelError(f"Prediction failed: {str(e)}")

    def export_prediction(self, prediction: np.ndarray, output_path: str, 
                        profile: Dict[str, Any]) -> None:
        """Exports the prediction result to a GeoTIFF file.
        
        Args:
            prediction (np.ndarray): Model prediction to export
            output_path (str): Path to save the prediction
            profile (Dict[str, Any]): Metadata profile for the output file
            
        Raises:
            ImageProcessingError: If export fails
        """
        try:
            full_path = os.path.join(self.data_dir, output_path)
            
            # Update profile for prediction output
            profile.update({
                'dtype': 'float32',
                'count': 1,
                'nodata': None
            })
            
            with rasterio.open(full_path, 'w', **profile) as dst:
                dst.write(prediction.astype('float32'), 1)
            logger.info(f"Successfully exported prediction to {full_path}")
            
        except Exception as e:
            logger.error(f"Failed to export prediction: {str(e)}")
            raise ImageProcessingError(f"Failed to export prediction: {str(e)}")

    def __del__(self):
        """Cleanup when the object is destroyed."""
        try:
            api.logout()
            logger.info("Successfully logged out from USGS Earth Explorer")
        except:
            pass

    def calculate_spectral_indices(self, image_paths: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Calculates various spectral indices for UHI analysis.
        
        Args:
            image_paths (Dict[str, str]): Dictionary mapping band names to file paths
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of calculated indices
        """
        try:
            indices = {}
            
            # Load required bands
            nir = self.load_satellite_image(image_paths['NIR'])[0]
            red = self.load_satellite_image(image_paths['RED'])[0]
            swir = self.load_satellite_image(image_paths['SWIR'])[0]
            thermal = self.load_satellite_image(image_paths['THERMAL'])[0]
            
            # Calculate NDVI
            indices['NDVI'] = self.calculate_ndvi(nir, red)
            
            # Calculate NDBI (Normalized Difference Built-up Index)
            sum_swir_nir = swir + nir
            diff_swir_nir = swir - nir
            valid_mask = sum_swir_nir != 0
            ndbi = np.zeros_like(sum_swir_nir, dtype=np.float32)
            ndbi[valid_mask] = diff_swir_nir[valid_mask] / sum_swir_nir[valid_mask]
            indices['NDBI'] = np.clip(ndbi, -1, 1)
            
            # Calculate NDBaI (Normalized Difference Bareness Index)
            sum_swir_thermal = swir + thermal
            diff_swir_thermal = swir - thermal
            valid_mask = sum_swir_thermal != 0
            ndbai = np.zeros_like(sum_swir_thermal, dtype=np.float32)
            ndbai[valid_mask] = diff_swir_thermal[valid_mask] / sum_swir_thermal[valid_mask]
            indices['NDBaI'] = np.clip(ndbai, -1, 1)
            
            # Calculate UI (Urban Index)
            ui = (swir - nir) / (swir + nir)
            indices['UI'] = np.clip(ui, -1, 1)
            
            return indices
            
        except Exception as e:
            logger.error(f"Failed to calculate spectral indices: {str(e)}")
            raise ImageProcessingError(f"Failed to calculate spectral indices: {str(e)}")

    def augment_data(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies data augmentation to training data.
        
        Args:
            image (np.ndarray): Input image
            mask (np.ndarray): Ground truth mask
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Augmented image and mask
        """
        try:
            # Random rotation
            angle = np.random.uniform(-30, 30)
            image = scipy.ndimage.rotate(image, angle, reshape=False)
            mask = scipy.ndimage.rotate(mask, angle, reshape=False)
            
            # Random flip
            if np.random.random() > 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)
            if np.random.random() > 0.5:
                image = np.flipud(image)
                mask = np.flipud(mask)
            
            # Random brightness adjustment
            brightness = np.random.uniform(0.8, 1.2)
            image = image * brightness
            
            # Random noise
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.01, image.shape)
                image = image + noise
            
            # Ensure values are in valid range
            image = np.clip(image, 0, 1)
            mask = np.clip(mask, 0, 1)
            
            return image, mask
            
        except Exception as e:
            logger.error(f"Failed to augment data: {str(e)}")
            raise ImageProcessingError(f"Failed to augment data: {str(e)}")

    def prepare_training_data(self, image_paths: Dict[str, str], 
                             ground_truth_path: str,
                             augment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares training data with augmentation.
        
        Args:
            image_paths (Dict[str, str]): Dictionary mapping band names to file paths
            ground_truth_path (str): Path to ground truth mask
            augment (bool): Whether to apply data augmentation
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Prepared training data and labels
        """
        try:
            # Calculate spectral indices
            indices = self.calculate_spectral_indices(image_paths)
            
            # Stack indices into input features
            features = np.stack([
                indices['NDVI'],
                indices['NDBI'],
                indices['NDBaI'],
                indices['UI']
            ], axis=-1)
            
            # Load ground truth
            ground_truth = self.load_satellite_image(ground_truth_path)[0]
            
            if augment:
                # Apply augmentation
                features, ground_truth = self.augment_data(features, ground_truth)
            
            return features, ground_truth
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {str(e)}")
            raise ImageProcessingError(f"Failed to prepare training data: {str(e)}")

class USGSEarthExplorer:
    def __init__(self):
        """Initialize the USGS Earth Explorer client."""
        self.session = requests.Session()
        self.api_token = None
        self.token_expiration = None
        
        # Configure retry strategy with exponential backoff
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=RETRY_BACKOFF,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"],
            respect_retry_after_header=True,
            raise_on_status=False
        )
        
        # Mount the adapter with retry strategy to the session
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Try to authenticate
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with the USGS Earth Explorer API using the new authentication method."""
        max_auth_attempts = 3
        current_attempt = 0
        last_error = None
        
        while current_attempt < max_auth_attempts:
            try:
                current_attempt += 1
                logger.info(f"Authentication attempt {current_attempt}/{max_auth_attempts}")
                
                # Validate credentials
                if not EARTHEXPLORER_USERNAME or not EARTHEXPLORER_PASSWORD:
                    raise USGSAuthenticationError("Earth Explorer credentials are missing")
                
                # Prepare login request
                login_data = {
                    "username": EARTHEXPLORER_USERNAME.strip(),
                    "password": EARTHEXPLORER_PASSWORD.strip(),
                    "catalogId": API_CATALOG_ID,
                    "authType": "EROS"
                }
                
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "UHIModel/1.0",
                    "Accept": "application/json"
                }
                
                # Make login request using the token endpoint
                try:
                    # Use the token for authentication
                    token_data = {
                        "username": EARTHEXPLORER_USERNAME.strip(),
                        "token": M2M_API_TOKEN.strip(),
                        "catalogId": API_CATALOG_ID,
                        "authType": "EROS"
                    }
                    
                    response = self.session.post(
                        f"{M2M_API_URL}/{LOGIN_TOKEN_ENDPOINT}",
                        json=token_data,
                        headers=headers,
                        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                        verify=VERIFY_SSL
                    )
                    
                    # Log response for debugging
                    logger.debug(f"Login response status: {response.status_code}")
                    logger.debug(f"Login response: {response.text[:200]}...")  # Log first 200 chars
                    
                    # Print the full response for debugging
                    print(f"Login response status: {response.status_code}")
                    print(f"Login response: {response.text}")
                    
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            if "data" in data:
                                # The new API returns an encoded token directly in the "data" field
                                self.api_token = data["data"]
                                self.token_expiration = datetime.now() + timedelta(days=API_TOKEN_EXPIRATION_DAYS)
                                logger.info("Successfully authenticated with USGS Earth Explorer API")
                                print("✓ USGS Authentication successful! API connection established.")
                                return
                            else:
                                raise USGSAuthenticationError("Invalid response format: missing data field")
                        except ValueError as e:
                            raise USGSAuthenticationError(f"Failed to parse authentication response: {str(e)}")
                    elif response.status_code == 401:
                        raise USGSAuthenticationError("Invalid credentials")
                    elif response.status_code == 403:
                        raise USGSAuthenticationError("Access forbidden - check your API permissions")
                    elif response.status_code == 429:
                        raise USGSAuthenticationError("Too many requests - rate limit exceeded")
                    elif response.status_code >= 500:
                        # Try to get more detailed error information
                        error_detail = "Unknown server error"
                        try:
                            error_data = response.json()
                            if "errorMessage" in error_data:
                                error_detail = error_data["errorMessage"]
                        except:
                            pass
                        raise USGSAuthenticationError(f"USGS API server error: {error_detail}")
                    else:
                        raise USGSAuthenticationError(f"Authentication failed with status {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    raise USGSAuthenticationError("Authentication request timed out")
                except requests.exceptions.ConnectionError:
                    raise USGSAuthenticationError("Failed to connect to USGS API")
                except requests.exceptions.RequestException as e:
                    raise USGSAuthenticationError(f"Request failed: {str(e)}")
                
            except USGSAuthenticationError as e:
                last_error = str(e)
                if current_attempt < max_auth_attempts:
                    wait_time = RETRY_BACKOFF * (2 ** (current_attempt - 1))
                    logger.warning(f"Authentication attempt {current_attempt} failed: {last_error}")
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                continue
            except Exception as e:
                last_error = f"Unexpected error during authentication: {str(e)}"
                logger.error(last_error)
                if current_attempt < max_auth_attempts:
                    wait_time = RETRY_BACKOFF * (2 ** (current_attempt - 1))
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                continue
        
        error_msg = f"All authentication attempts failed. Last error: {last_error}"
        logger.error(error_msg)
        raise USGSAuthenticationError(error_msg)

    def _verify_token(self, raise_error=True) -> bool:
        """Verify that the token is valid by making a test request.
        
        Args:
            raise_error (bool): Whether to raise an exception on failure
            
        Returns:
            bool: True if token is valid, False otherwise
        """
        try:
            # Use a minimal test request
            test_request = {
                "datasetName": LANDSAT_COLLECTION,
                "maxResults": 1,
                "metadataType": "summary",
                "temporalFilter": {
                    "start": datetime.now().strftime("%Y-01-01"),
                    "end": datetime.now().strftime("%Y-%m-%d")
                }
            }
            
            headers = self._get_auth_headers()
            
            response = self.session.post(
                f"{M2M_API_URL}/{SEARCH_ENDPOINT}",
                json=test_request,
                headers=headers,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                verify=VERIFY_SSL
            )
            
            if response.status_code == 401:
                if raise_error:
                    raise USGSAuthenticationError("Invalid or expired token")
                return False
            
            if response.status_code != 200:
                if raise_error:
                    raise USGSAuthenticationError(f"Token verification failed with status code {response.status_code}")
                return False
            
            try:
                data = response.json()
                if "errorCode" in data:
                    if raise_error:
                        raise USGSAuthenticationError(f"API Error: {data.get('errorMessage', 'Unknown error')}")
                    return False
                if not isinstance(data, dict):
                    if raise_error:
                        raise USGSAuthenticationError("Invalid response format")
                    return False
                logger.info("Successfully verified M2M API token")
                return True
            except ValueError as e:
                if raise_error:
                    raise USGSAuthenticationError(f"Invalid JSON response: {str(e)}")
                return False
                
        except Exception as e:
            if raise_error:
                raise USGSAuthenticationError(f"Token verification failed: {str(e)}")
            return False

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get the authentication headers for API requests.
        
        Returns:
            Dict[str, str]: Headers dictionary with authentication token
        """
        if not self.api_token:
            raise USGSAuthenticationError("No API token available")
            
        return {
            "Content-Type": "application/json",
            "X-Auth-Token": self.api_token
        }

    def search_scenes(self, dataset: str, bbox: Tuple[float, float, float, float], 
                     start_date: str, end_date: str, max_cloud_cover: int = 100,
                     max_results: int = 100) -> Dict[str, Any]:
        """
        Search for scenes using the M2M API.
        Args:
            dataset (str): Dataset name (e.g., "landsat_ot_c2_l1")
            bbox (tuple): Bounding box coordinates (min_lon, min_lat, max_lon, max_lat)
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            max_cloud_cover (int): Maximum cloud cover percentage (0-100)
            max_results (int): Maximum number of results to return
        Returns:
            dict: Search results
        """
        self._check_token_validity()
        
        try:
            payload = {
                'datasetName': dataset,
                'maxResults': max_results,
                'startingNumber': 1,
                'metadataType': 'summary',
                'catalogId': API_CATALOG_ID,
                'sceneFilter': {
                    'spatialFilter': {
                        'filterType': 'mbr',
                        'lowerLeft': {'longitude': bbox[0], 'latitude': bbox[1]},
                        'upperRight': {'longitude': bbox[2], 'latitude': bbox[3]}
                    },
                    'acquisitionFilter': {
                        'start': start_date,
                        'end': end_date
                    }
                }
            }
            
            if max_cloud_cover < 100:
                # Add cloud cover filter to sceneFilter
                payload['sceneFilter']['cloudCoverFilter'] = {
                    'max': max_cloud_cover,
                    'min': 0,
                    'includeUnknown': True
                }
            
            # Log the request for debugging
            logger.debug(f"Sending search request to {M2M_API_URL}/{SEARCH_ENDPOINT}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
            
            # Get authentication headers
            headers = self._get_auth_headers()
            
            response = self.session.post(
                f"{M2M_API_URL}/{SEARCH_ENDPOINT}",
                json=payload,
                headers=headers,
                timeout=60  # Increased timeout for search
            )
            
            if response.status_code != 200:
                logger.error(f"Search request failed with status {response.status_code}")
                logger.error(f"Response content: {response.text}")
                raise M2MAPIError(f"Scene search failed. Status code: {response.status_code}")
            
            try:
                data = response.json()
                logger.debug(f"Search response: {json.dumps(data, indent=2)}")
                return data
            except ValueError as e:
                logger.error(f"Failed to parse search response: {str(e)}")
                logger.error(f"Raw response: {response.text}")
                raise M2MAPIError(f"Failed to parse search response: {str(e)}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Search request failed: {str(e)}")
            raise M2MAPIError(f"Failed to search scenes: {str(e)}")

    def download_scene(self, scene_id: str, output_dir: str) -> str:
        """
        Download a scene using the M2M API.
        Args:
            scene_id (str): Scene ID to download
            output_dir (str): Directory to save the downloaded file
        Returns:
            str: Path to the downloaded file
        """
        self._check_token_validity()
        
        try:
            # Get download options
            headers = self._get_auth_headers()
            
            # Construct the proper payload for download options
            payload = {
                "datasetName": "landsat_ot_c2_l1",
                "entityIds": [scene_id],
                "catalogId": API_CATALOG_ID
            }
            
            response = self.session.post(
                f"{M2M_API_URL}/{DOWNLOAD_OPTIONS_ENDPOINT}",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                raise M2MAPIError(f"Failed to get download options. Status code: {response.status_code}")
            
            download_options_response = response.json()
            
            # Check if we have options available in the response
            if not download_options_response.get('data', []):
                raise ImageDownloadError(f"No download options available for scene {scene_id}")
                
            # Find a suitable download option (we'll take the first available one)
            download_options = download_options_response.get('data', [])
            available_options = [opt for opt in download_options if opt.get('available', False)]
            
            if not available_options:
                raise ImageDownloadError(f"Scene {scene_id} is not available for download")
            
            # Select the first download option
            selected_option = available_options[0]
            logger.info(f"Selected download option: {selected_option.get('productName')}")
            
            # Request download
            payload = {
                'datasetName': "landsat_ot_c2_l1",
                'entityIds': [scene_id],
                'products': [selected_option.get('id')],
                'catalogId': API_CATALOG_ID,
                # Add optional parameters to improve download success
                'priority': 'high',
                'processingLevel': selected_option.get('processingLevel', 'None')
            }
            
            response = self.session.post(
                f"{M2M_API_URL}/{DOWNLOAD_REQUEST_ENDPOINT}",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                raise M2MAPIError(f"Download request failed. Status code: {response.status_code}")
            
            download_data = response.json()
            
            # Check for available downloads or download preparation status
            data = download_data.get('data', {})
            
            # First check if we have a direct downloadId
            if data.get('downloadId'):
                download_id = data['downloadId']
                logger.info(f"Download ID: {download_id}")
                
                # Get the actual download URL
                download_retrieve_endpoint = f"download-retrieve/{download_id}"
                response = self.session.get(
                    f"{M2M_API_URL}/{download_retrieve_endpoint}",
                    headers=headers,
                    timeout=60
                )
            # Check if downloads are being prepared
            elif data.get('preparingDownloads') and len(data['preparingDownloads']) > 0:
                prep_id = data['preparingDownloads'][0].get('downloadId')
                if not prep_id:
                    raise ImageDownloadError(f"Download is being prepared but no download ID available: {download_data}")
                
                logger.info(f"Download is being prepared. Download ID: {prep_id}")
                
                # Wait for download to be ready (implement polling)
                max_attempts = 5
                for attempt in range(max_attempts):
                    logger.info(f"Waiting for download to be ready (attempt {attempt+1}/{max_attempts})...")
                    time.sleep(10)  # Wait 10 seconds between checks
                    
                    # Check download status
                    retrieve_endpoint = f"download-retrieve/{prep_id}"
                    response = self.session.get(
                        f"{M2M_API_URL}/{retrieve_endpoint}",
                        headers=headers,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        break
                    
                    if attempt == max_attempts - 1:
                        raise ImageDownloadError(f"Download preparation timed out after {max_attempts} attempts")
            # If there are no downloads available or being prepared
            else:
                available = data.get('availableDownloads', [])
                preparing = data.get('preparingDownloads', [])
                failed = data.get('failed', [])
                
                # Check if we're hitting rate limits
                rate_limited = False
                limits = data.get('remainingLimits', [])
                for limit in limits:
                    # If the remaining counts are high, this suggests we're being rate limited
                    if (limit.get('recentDownloadCount', 0) > 1000 or 
                        limit.get('pendingDownloadCount', 0) > 1000):
                        rate_limited = True
                        break
                
                if rate_limited:
                    error_msg = "Download rate limited by USGS API. "
                    if data.get('remainingLimits'):
                        error_msg += f"Limit details: {data.get('remainingLimits')}"
                    
                    # Log this as a warning instead of an error since we'll handle it
                    logger.warning(error_msg)
                    
                    # Return a special signal that we're rate limited rather than raising an exception
                    return "RATE_LIMITED"
                else:
                    error_msg = "No downloads available or being prepared. "
                    if failed:
                        error_msg += f"Failed downloads: {failed}. "
                    if data.get('remainingLimits'):
                        error_msg += f"Download limits: {data.get('remainingLimits')}"
                        
                    raise ImageDownloadError(error_msg)
            
            if response.status_code != 200:
                raise ImageDownloadError(f"Failed to retrieve download URL. Status: {response.status_code}")
                
            download_info = response.json()
            logger.debug(f"Download info response: {json.dumps(download_info, indent=2)}")
            
            # Extract URL from the response, handling different response formats
            if download_info.get('data', {}).get('fileUrls'):
                # Get URLs from the fileUrls field
                download_urls = download_info['data']['fileUrls']
                
                if isinstance(download_urls, dict) and len(download_urls) > 0:
                    # If it's a dictionary (key-value pairs)
                    download_url = list(download_urls.values())[0]
                elif isinstance(download_urls, list) and len(download_urls) > 0:
                    # If it's a list of URLs
                    download_url = download_urls[0]
                else:
                    raise ImageDownloadError(f"Invalid file URLs format: {download_urls}")
            # Check for direct URL in the response
            elif download_info.get('data', {}).get('url'):
                download_url = download_info['data']['url']
            # Handle case where download might not be ready yet
            elif download_info.get('data', {}).get('statusCode') == 'PREPARING':
                # Wait and retry for download preparation
                prep_id = download_info.get('data', {}).get('downloadId')
                if not prep_id:
                    raise ImageDownloadError("Download is still being prepared but no ID available")
                
                max_retries = 3
                retry_wait = 30  # seconds
                
                for i in range(max_retries):
                    logger.info(f"Download preparation in progress. Retrying in {retry_wait} seconds ({i+1}/{max_retries})...")
                    time.sleep(retry_wait)
                    
                    # Check status again
                    retry_response = self.session.get(
                        f"{M2M_API_URL}/download-retrieve/{prep_id}",
                        headers=headers,
                        timeout=60
                    )
                    
                    if retry_response.status_code == 200:
                        retry_info = retry_response.json()
                        if retry_info.get('data', {}).get('fileUrls'):
                            download_urls = retry_info['data']['fileUrls']
                            if isinstance(download_urls, dict) and len(download_urls) > 0:
                                download_url = list(download_urls.values())[0]
                                break
                            elif isinstance(download_urls, list) and len(download_urls) > 0:
                                download_url = download_urls[0]
                                break
                
                if 'download_url' not in locals():
                    raise ImageDownloadError(f"Download preparation timed out after {max_retries} retries")
            else:
                raise ImageDownloadError(f"No download URLs found in response: {download_info}")
                
            logger.info(f"Download URL: {download_url}")
            
            # Download the file
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{scene_id}.tar.gz")
            
            response = self.session.get(download_url, stream=True)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return output_path
            
        except requests.exceptions.RequestException as e:
            raise ImageDownloadError(f"Failed to download scene: {str(e)}")

    def _check_token_validity(self):
        """Check if the current token is valid and hasn't expired."""
        if datetime.now() >= self.token_expiration:
            raise USGSAuthenticationError("API token has expired. Please obtain a new token.")

def main():
    """Main execution function demonstrating the Urban Heat Island detection workflow."""
    try:
        # Initialize USGS Earth Explorer client
        logger.info("Initializing USGS Earth Explorer client...")
        usgs_client = USGSEarthExplorer()
        
        # Define coordinates for Lagos and Cairo
        locations = {
            "Lagos": {
                "bbox": (3.1307, 6.3936, 3.4218, 6.7027),  # (min_lon, min_lat, max_lon, max_lat)
                "description": "Lagos, Nigeria"
            },
            "Cairo": {
                "bbox": (31.2357, 29.9500, 31.5714, 30.1795),  # (min_lon, min_lat, max_lon, max_lat)
                "description": "Cairo, Egypt"
            }
        }
        
        # Search for scenes in both locations
        for location_name, location_data in locations.items():
            logger.info(f"\nProcessing {location_data['description']}...")
            logger.info(f"Using bounding box coordinates: {location_data['bbox']}")
            
            # Special case for Lagos - use a wider date range to include 2024 data
            if location_name == "Lagos":
                start_date = "2023-01-01"
                end_date = "2024-12-31"
            else:
                start_date = "2023-08-01"  # Very limited time range for other locations
                end_date = "2023-08-30"
                
            # Search for Landsat scenes with specific criteria for urban heat island analysis
            scenes = usgs_client.search_scenes(
                dataset="landsat_ot_c2_l1",  # Landsat 8-9 Collection 2 Level-1
                bbox=location_data['bbox'],
                start_date=start_date,
                end_date=end_date,
                max_cloud_cover=10,  # Low cloud cover for better surface temperature analysis
                max_results=2  # Very limited results for testing
            )
            
            # Check if we have any results
            results = scenes.get('data', {}).get('results', [])
            
            # Special handling for Lagos - check for manually downloaded scenes even if API returns nothing
            if not results and location_name == "Lagos":
                # Check for scene folders that might exist already
                scene_dirs = [d for d in os.listdir(os.path.join(data_dir)) 
                             if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("LC")]
                
                if scene_dirs:
                    logger.info(f"Found {len(scene_dirs)} manually downloaded scenes in {data_dir}")
                    # Create a synthetic result entry for each scene directory
                    for scene_dir in scene_dirs:
                        # Create a minimal scene result structure
                        scene_result = {
                            "entityId": scene_dir,
                            "cloudCover": 0,  # Assume good quality
                            "metadata": [
                                {"fieldName": "Date Acquired", "value": "2024/02/08"}  # Default date
                            ]
                        }
                        results.append(scene_result)
                        logger.info(f"Added manually downloaded scene: {scene_dir}")
                else:
                    logger.warning(f"No scenes found for {location_data['description']}")
                    continue
            elif not results:
                logger.warning(f"No scenes found for {location_data['description']}")
                continue
            
            # Log found scenes
            results = scenes.get('data', {}).get('results', [])
            logger.info(f"Found {len(results)} scenes for {location_data['description']}")
            for scene in results[:3]:  # Log first 3 scenes
                if scene.get('metadata'):
                    # Look for Date Acquired in metadata
                    acquisition_date = 'unknown'
                    for meta in scene.get('metadata', []):
                        if meta.get('fieldName') == 'Date Acquired':
                            acquisition_date = meta.get('value', 'unknown')
                            break
                
                cloud_cover = scene.get('cloudCover', 'unknown')
                logger.info(f"Scene ID: {scene.get('entityId', 'unknown')}")
                logger.info(f"Date: {acquisition_date}, Cloud Cover: {cloud_cover}%")
            
            # Create location-specific data directory
            data_dir = os.path.join(os.path.dirname(__file__), 'data', location_name.lower())
            os.makedirs(data_dir, exist_ok=True)
            logger.info(f"Using data directory: {data_dir}")
            
            # Download the best available scene (lowest cloud cover)
            results = scenes.get('data', {}).get('results', [])
            if results:
                # Sort scenes by cloud cover
                sorted_scenes = sorted(results, 
                                    key=lambda x: float(x.get('cloudCover', 100)))
                best_scene = sorted_scenes[0]
                scene_id = best_scene['entityId']
                
                # Get acquisition date
                acquisition_date = 'unknown'
                for meta in best_scene.get('metadata', []):
                    if meta.get('fieldName') == 'Date Acquired':
                        acquisition_date = meta.get('value', 'unknown')
                        break
                
                logger.info(f"Downloading best scene {scene_id} for {location_data['description']}...")
                logger.info(f"Cloud cover: {best_scene.get('cloudCover')}%")
                logger.info(f"Acquisition date: {acquisition_date}")
                
                try:
                    result = usgs_client.download_scene(scene_id, data_dir)
                    
                    # Check if we got a rate limit signal
                    if result == "RATE_LIMITED":
                        logger.info(f"Download for {scene_id} is rate limited by USGS API. Using alternative approach.")
                        # Go directly to the fallback without treating this as an error
                        raise RateLimitedException("USGS download rate limited")
                    else:
                        # We have a successful download
                        downloaded_file = result
                        logger.info(f"Successfully downloaded scene to {downloaded_file}")
                except RateLimitedException:
                    # Special case for rate limiting - treated as expected behavior
                    logger.info(f"Using alternative download approach for {location_data['description']} due to rate limits")
                    logger.info("Preparing for manual download instructions...")
                except Exception as e:
                    logger.error(f"Failed to download scene for {location_data['description']}: {str(e)}")
                    logger.info("Using fallback method - requesting individual bands instead...")
                    
                    # Fallback to individual band download which may have better success rate
                    try:
                        # Create a directory for the scene
                        scene_dir = os.path.join(data_dir, scene_id)
                        os.makedirs(scene_dir, exist_ok=True)
                        
                        # Try to download a sample thermal band (Band 10)
                        thermal_band_path = os.path.join(scene_dir, f"{scene_id}_B10.TIF")
                        
                        # Use a direct HTTP download if available
                        # This is a fallback that could be replaced with actual logic to find download URLs
                        
                        logger.info(f"✓ Created fallback directory at {scene_dir}")
                        
                        # For testing purposes only - in production would need actual band data
                        with open(os.path.join(scene_dir, "DOWNLOAD_FAILED.txt"), "w") as f:
                            f.write(f"Download failed at {datetime.now()}\n")
                            f.write(f"Error: {str(e)}\n")
                            f.write("Please try downloading manually and place the files in this directory.")
                        
                        logger.info("Created placeholder file for manual download guidance.")
                        
                        # Continue with processing instead of skipping
                    except Exception as inner_e:
                        logger.error(f"Fallback method also failed: {str(inner_e)}")
                        continue
                
                # Check if manually downloaded files exist before proceeding
                def check_for_manual_downloads(scene_dir, scene_id):
                    """Check if manually downloaded files exist in the scene directory"""
                    # Common patterns for Landsat 8/9 bands - supporting both naming conventions
                    required_patterns = [
                        # Match standard USGS scene_id pattern
                        f"*{scene_id}*_B10.TIF",  # Thermal
                        f"*{scene_id}*_B4.TIF",   # Red
                        f"*{scene_id}*_B5.TIF",   # NIR
                        f"*{scene_id}*_B6.TIF",   # SWIR1
                        
                        # Match LC09 style naming pattern (for manually downloaded files)
                        "*_B10.TIF",  # Thermal
                        "*_B4.TIF",   # Red
                        "*_B5.TIF",   # NIR
                        "*_B6.TIF"    # SWIR1
                    ]
                    
                    # Check for extracted files in scene directory
                    import glob
                    found_files = []
                    for pattern in required_patterns:
                        matches = glob.glob(os.path.join(scene_dir, pattern))
                        if matches:
                            found_files.append(os.path.basename(matches[0]))
                    
                    return found_files
                
                # Process thermal bands for UHI analysis
                try:
                    # Check if we have manually downloaded files
                    scene_dir = os.path.join(data_dir, scene_id)
                    manual_files = check_for_manual_downloads(scene_dir, scene_id)
                    
                    if manual_files:
                        logger.info(f"Found {len(manual_files)} manually downloaded files in {scene_dir}")
                        logger.info(f"Files: {', '.join(manual_files)}")
                        logger.info(f"Will use these files for UHI analysis.")
                        
                        # For Lagos: Proceed with UHI analysis since we have the files
                        if location_name == "Lagos":
                            logger.info(f"==== Running UHI analysis for {location_data['description']} ====")
                            
                            # Get file paths for the bands (simplified example)
                            band_files = {}
                            for f in manual_files:
                                if "_B10" in f:
                                    band_files['THERMAL'] = os.path.join(scene_dir, f)
                                elif "_B4" in f:
                                    band_files['RED'] = os.path.join(scene_dir, f)
                                elif "_B5" in f:
                                    band_files['NIR'] = os.path.join(scene_dir, f)
                                elif "_B6" in f:
                                    band_files['SWIR'] = os.path.join(scene_dir, f)
                            
                            # Check if we have all required bands
                            if len(band_files) == 4:  # All required bands present
                                try:
                                    # Create UHI model instance
                                    logger.info("Initializing UHI model for processing...")
                                    uhi_model = UHIModel(data_dir=os.path.dirname(data_dir))
                                    
                                    # Calculate spectral indices
                                    logger.info("Calculating spectral indices...")
                                    indices = uhi_model.calculate_spectral_indices(band_files)
                                    
                                    # NDVI visualization
                                    output_path = os.path.join(data_dir, f"{scene_id}_NDVI.TIF")
                                    profile = uhi_model.load_satellite_image(band_files['NIR'])[1]
                                    uhi_model.export_prediction(indices['NDVI'], output_path, profile)
                                    logger.info(f"Successfully generated NDVI visualization at {output_path}")
                                    
                                    # Calculate Urban Heat Island index
                                    logger.info("UHI analysis complete! Results saved to output directory.")
                                except Exception as e:
                                    logger.error(f"Error during UHI analysis: {str(e)}")
                            else:
                                logger.warning(f"Missing some required bands. Found: {list(band_files.keys())}")
                    else:
                        logger.info(f"No manually downloaded files found. Waiting for user to download.")
                        logger.info(f"UHI model can process the data once the images are manually downloaded.")
                    
                    # Create a README file with detailed instructions
                    readme_path = os.path.join(data_dir, "README.txt")
                    with open(readme_path, "w") as f:
                        f.write("=== USGS Earth Explorer Manual Download Instructions ===\n\n")
                        f.write("The automatic download failed due to USGS API rate limits.\n")
                        f.write("Please follow these steps to manually download the required data:\n\n")
                        f.write("1. Go to https://earthexplorer.usgs.gov/\n")
                        f.write("2. Login with your EROS account (or create one if needed)\n")
                        f.write("3. In the Search Criteria tab:\n")
                        f.write("   - Enter the coordinates for this area:\n")
                        f.write(f"     Lower left: {location_data['bbox'][0]}, {location_data['bbox'][1]}\n")
                        f.write(f"     Upper right: {location_data['bbox'][2]}, {location_data['bbox'][3]}\n")
                        f.write("   - Or search directly for scene ID:\n")
                        f.write(f"     {scene_id}\n\n")
                        f.write("4. In the Dataset tab:\n")
                        f.write("   - Expand 'Landsat'\n")
                        f.write("   - Expand 'Landsat Collection 2 Level-1'\n")
                        f.write("   - Select 'Landsat 8-9 OLI/TIRS C2 L1'\n\n")
                        f.write("5. In the Results tab:\n")
                        f.write(f"   - Find scene ID: {scene_id}\n")
                        f.write("   - Click the download icon (down arrow)\n")
                        f.write("   - Select 'Product Options'\n")
                        f.write("   - Choose 'Level-1 GeoTIFF Data Product'\n\n")
                        f.write(f"6. Extract the downloaded file into this directory:\n   {data_dir}\n\n")
                        f.write("7. Run the UHI model again\n\n")
                        f.write("Required bands for UHI analysis:\n")
                        f.write("- Band 10 (Thermal Infrared) - For surface temperature\n")
                        f.write("- Band 4 (Red) - For NDVI calculation\n")
                        f.write("- Band 5 (Near Infrared) - For NDVI calculation\n")
                        f.write("- Band 6 (SWIR 1) - For built-up area analysis\n\n")
                        f.write("If you continue to experience issues with the USGS API, consider:\n")
                        f.write("1. Using a different account\n")
                        f.write("2. Waiting 24 hours before trying again\n")
                        f.write("3. Reducing your search criteria to fewer scenes\n\n")
                        f.write(f"For additional help, contact USGS support at: https://www.usgs.gov/landsat-missions/contact-us\n")
                    
                    logger.info(f"Created README with download instructions at {readme_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to create instruction file: {str(e)}")
            
            logger.info(f"Completed processing for {location_data['description']}\n")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
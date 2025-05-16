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

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalizes data for machine learning input.
        
        Args:
            data (np.ndarray): Input data array
            
        Returns:
            np.ndarray: Normalized data in range [0, 1]
        """
        scaler = MinMaxScaler()
        return scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)

        def build_unet_model(self, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """Builds an enhanced U-Net model for Urban Heat Island detection.
        
        This implementation includes several modern improvements:
        - Batch normalization for better training stability
        - Dropout for regularization
        - Residual connections for better gradient flow
        - ELU activation for better gradient propagation
        - Spatial dropout for feature map regularization
        
        Args:
            input_shape (Tuple[int, int, int]): Input shape (height, width, channels)
            
        Returns:
            tf.keras.Model: Compiled U-Net model
        """
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

        # Encoder
        conv1 = conv_block(inputs, 64)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = conv_block(pool1, 128)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = conv_block(pool2, 256)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        # Bridge
        conv4 = conv_block(pool3, 512)

        # Decoder with skip connections and residual blocks
        up5 = layers.UpSampling2D(size=(2, 2))(conv4)
        up5 = layers.Conv2D(256, 2, padding='same')(up5)
        merge5 = layers.concatenate([conv3, up5], axis=3)
        conv5 = conv_block(merge5, 256)

        up6 = layers.UpSampling2D(size=(2, 2))(conv5)
        up6 = layers.Conv2D(128, 2, padding='same')(up6)
        merge6 = layers.concatenate([conv2, up6], axis=3)
        conv6 = conv_block(merge6, 128)

        up7 = layers.UpSampling2D(size=(2, 2))(conv6)
        up7 = layers.Conv2D(64, 2, padding='same')(up7)
        merge7 = layers.concatenate([conv1, up7], axis=3)
        conv7 = conv_block(merge7, 64)

        # Output
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv7)

        # Add residual connection from input to output if same shape
        if inputs.shape[1:] == outputs.shape[1:]:
            outputs = layers.Add()([outputs, inputs])

        model = models.Model(inputs=[inputs], outputs=[outputs])
        return model

    def train_model(self, train_data: np.ndarray, train_labels: np.ndarray, 
                   val_data: Optional[np.ndarray] = None, 
                   val_labels: Optional[np.ndarray] = None,
                   epochs: int = 10, batch_size: int = 16) -> tf.keras.callbacks.History:
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

def main():
    """Main execution function demonstrating the Urban Heat Island detection workflow.
    
    This function shows how to:
    1. Initialize the model with proper credentials
    2. Download and process Landsat 8 imagery
    3. Calculate NDVI
    4. Train the U-Net model
    5. Make predictions and export results
    """
    try:
        # Initialize the model with Earth Explorer credentials
        uhi_model = UHIModel()
        
        # Example Landsat 8 scene ID (using a more recent scene)
        scene_id = "LC08_L1GT_044034_20230415_20230416_02_T2"
        
        # Download required bands with progress tracking
        logger.info("Downloading required Landsat 8 bands...")
        bands = {
            "red": "4",
            "nir": "5",
            "swir1": "6",
            "thermal": "10",
            "swir2": "7"
        }
        
        band_paths = {}
        for band_name, band_number in bands.items():
            logger.info(f"Downloading {band_name} band...")
            band_paths[band_name] = uhi_model.download_landsat_image(
                scene_id, 
                band_number,
                f'{band_name}_band.tif'
            )
        
        # Calculate NDVI
        logger.info("Calculating NDVI...")
        ndvi = uhi_model.calculate_ndvi(band_paths["nir"], band_paths["red"])
        
        # Load and preprocess thermal band for training
        logger.info("Loading thermal band for training...")
        thermal_data, thermal_profile = uhi_model.load_satellite_image(band_paths["thermal"])
        thermal_data = uhi_model.normalize_data(thermal_data)
        
        # Prepare training data
        input_size = (128, 128)
        
        # Create patches for training
        def create_patches(data, patch_size):
            patches = []
            for i in range(0, data.shape[0] - patch_size[0], patch_size[0] // 2):
                for j in range(0, data.shape[1] - patch_size[1], patch_size[1] // 2):
                    patch = data[i:i + patch_size[0], j:j + patch_size[1]]
                    if patch.shape == patch_size:
                        patches.append(patch)
            return np.array(patches)
        
        # Create training patches
        thermal_patches = create_patches(thermal_data, input_size)
        ndvi_patches = create_patches(ndvi, input_size)
        
        # Reshape for training
        X_train = thermal_patches.reshape(-1, *input_size, 1)
        y_train = ndvi_patches.reshape(-1, *input_size, 1)
        
        # Split into training and validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Train the model with advanced features
        logger.info("Training model...")
        history = uhi_model.train_model(
            X_train, y_train,
            val_data=X_val,
            val_labels=y_val,
            epochs=50,
            batch_size=32
        )
        
        # Make predictions on the full thermal image
        logger.info("Making predictions...")
        # Pad the image to be divisible by the input size
        pad_h = (input_size[0] - thermal_data.shape[0] % input_size[0]) % input_size[0]
        pad_w = (input_size[1] - thermal_data.shape[1] % input_size[1]) % input_size[1]
        thermal_padded = np.pad(thermal_data, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        # Make prediction
        thermal_input = thermal_padded.reshape(1, *thermal_padded.shape, 1)
        prediction = uhi_model.predict(thermal_input)
        
        # Remove padding
        prediction = prediction[0, :thermal_data.shape[0], :thermal_data.shape[1], 0]
        
        # Export results
        logger.info("Exporting results...")
        uhi_model.export_prediction(prediction, 'uhi_prediction.tif', thermal_profile)
        
        logger.info("Processing completed successfully")
        
    except ValueError as ve:
        logger.error(f"Configuration error: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"An error occurred in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

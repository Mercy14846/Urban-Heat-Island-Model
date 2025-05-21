import os
import logging
import rasterio
import numpy as np
from datetime import datetime
from rasterio.enums import Resampling
from sklearn.preprocessing import MinMaxScaler

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class OfflineUHIModel:
    """A simplified UHI model that works with local files only (no API connection needed)"""
    
    def __init__(self, data_dir="data"):
        """Initialize the offline UHI model with a data directory"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_satellite_image(self, file_path):
        """Loads satellite image using Rasterio with error handling."""
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
            raise Exception(f"Failed to load image: {str(e)}")
    
    def normalize_data(self, data):
        """Normalizes data to range [0, 1]"""
        try:
            # Handle data with all same values
            if np.max(data) == np.min(data):
                return np.zeros_like(data)
            
            # Normalize using min-max scaling
            normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
            return normalized
        except Exception as e:
            logger.error(f"Normalization error: {str(e)}")
            # Return original data if normalization fails
            return data
    
    def calculate_ndvi(self, nir_data, red_data):
        """Calculates NDVI from NIR and Red band data"""
        # Calculate NDVI with proper handling of edge cases
        sum_bands = nir_data + red_data
        diff_bands = nir_data - red_data
        
        # Handle division by zero and invalid values
        ndvi = np.zeros_like(sum_bands, dtype=np.float32)
        valid_mask = sum_bands != 0
        ndvi[valid_mask] = diff_bands[valid_mask] / sum_bands[valid_mask]
        
        # Clip values to valid NDVI range [-1, 1]
        ndvi = np.clip(ndvi, -1, 1)
        
        return self.normalize_data(ndvi)
    
    def calculate_ndbi(self, swir_data, nir_data):
        """Calculates NDBI (Normalized Difference Built-up Index)"""
        sum_bands = swir_data + nir_data
        diff_bands = swir_data - nir_data
        
        ndbi = np.zeros_like(sum_bands, dtype=np.float32)
        valid_mask = sum_bands != 0
        ndbi[valid_mask] = diff_bands[valid_mask] / sum_bands[valid_mask]
        
        return np.clip(ndbi, -1, 1)
    
    def calculate_urban_index(self, swir_data, nir_data):
        """Calculates Urban Index"""
        sum_bands = swir_data + nir_data
        diff_bands = swir_data - nir_data
        
        ui = np.zeros_like(sum_bands, dtype=np.float32)
        valid_mask = sum_bands != 0
        ui[valid_mask] = diff_bands[valid_mask] / sum_bands[valid_mask]
        
        return np.clip(ui, -1, 1)
    
    def calculate_spectral_indices(self, band_files):
        """Calculates various spectral indices"""
        try:
            indices = {}
            
            # Load bands
            nir_data, _ = self.load_satellite_image(band_files['NIR'])
            red_data, _ = self.load_satellite_image(band_files['RED'])
            swir_data, _ = self.load_satellite_image(band_files['SWIR'])
            thermal_data, _ = self.load_satellite_image(band_files['THERMAL'])
            
            # Ensure all bands have the same dimensions
            # Resample if needed (simplified for this script)
            
            # Calculate NDVI
            indices['NDVI'] = self.calculate_ndvi(nir_data, red_data)
            
            # Calculate NDBI
            indices['NDBI'] = self.calculate_ndbi(swir_data, nir_data)
            
            # Calculate Urban Index
            indices['UI'] = self.calculate_urban_index(swir_data, nir_data)
            
            # Normalize thermal data
            indices['THERMAL'] = self.normalize_data(thermal_data)
            
            return indices
            
        except Exception as e:
            logger.error(f"Failed to calculate spectral indices: {str(e)}")
            raise Exception(f"Failed to calculate spectral indices: {str(e)}")
    
    def export_prediction(self, prediction, output_path, profile):
        """Exports an array to a GeoTIFF file"""
        try:
            # Update profile for output
            profile.update({
                'dtype': 'float32',
                'count': 1,
                'nodata': None
            })
            
            # Make sure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(prediction.astype('float32'), 1)
            logger.info(f"Successfully exported data to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export data: {str(e)}")
            raise Exception(f"Failed to export data: {str(e)}")


def process_lagos_data():
    """Process the manually downloaded Lagos data offline"""
    try:
        # Define the scene ID and data directory
        scene_id = "LC91910552024039LGN00"
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, 'data', 'lagos')
        scene_dir = os.path.join(data_dir, scene_id)
        
        logger.info(f"Processing Lagos data in {scene_dir} (offline mode)")
        
        # Check if scene directory exists
        if not os.path.exists(scene_dir):
            logger.error(f"Directory not found: {scene_dir}")
            return
            
        # Find the band files
        import glob
        band_files = {}
        
        # Find files using the LC09 pattern
        thermal_files = glob.glob(os.path.join(scene_dir, "*B10.TIF"))
        red_files = glob.glob(os.path.join(scene_dir, "*B4.TIF"))
        nir_files = glob.glob(os.path.join(scene_dir, "*B5.TIF"))
        swir_files = glob.glob(os.path.join(scene_dir, "*B6.TIF"))
        
        # Assign band files if found
        if thermal_files:
            band_files['THERMAL'] = thermal_files[0]
            logger.info(f"Found Thermal band: {os.path.basename(thermal_files[0])}")
        
        if red_files:
            band_files['RED'] = red_files[0]
            logger.info(f"Found Red band: {os.path.basename(red_files[0])}")
            
        if nir_files:
            band_files['NIR'] = nir_files[0]
            logger.info(f"Found NIR band: {os.path.basename(nir_files[0])}")
            
        if swir_files:
            band_files['SWIR'] = swir_files[0]
            logger.info(f"Found SWIR band: {os.path.basename(swir_files[0])}")
        
        # Check if we have all required bands
        required_bands = ['THERMAL', 'RED', 'NIR', 'SWIR']
        missing_bands = [band for band in required_bands if band not in band_files]
        
        if missing_bands:
            logger.warning(f"Missing required bands: {missing_bands}")
            logger.warning("The UHI model requires Thermal (B10), Red (B4), NIR (B5), and SWIR (B6) bands.")
            return
            
        # Initialize offline UHI model
        logger.info("Initializing offline UHI model...")
        uhi_model = OfflineUHIModel(data_dir=base_dir)
        
        # Process the data
        logger.info("Calculating spectral indices for Urban Heat Island analysis...")
        indices = uhi_model.calculate_spectral_indices(band_files)
        
        # Get profile for output files
        _, profile = uhi_model.load_satellite_image(band_files['NIR'])
        
        # Save NDVI result
        ndvi_output = os.path.join(data_dir, f"{scene_id}_NDVI.TIF")
        uhi_model.export_prediction(indices['NDVI'], ndvi_output, profile)
        logger.info(f"Successfully saved NDVI visualization to {ndvi_output}")
        
        # Save NDBI result (Normalized Difference Built-up Index)
        ndbi_output = os.path.join(data_dir, f"{scene_id}_NDBI.TIF")
        uhi_model.export_prediction(indices['NDBI'], ndbi_output, profile)
        logger.info(f"Successfully saved NDBI visualization to {ndbi_output}")
        
        # Save Urban Index result
        ui_output = os.path.join(data_dir, f"{scene_id}_UI.TIF")
        uhi_model.export_prediction(indices['UI'], ui_output, profile)
        logger.info(f"Successfully saved Urban Index visualization to {ui_output}")
        
        # Create a simple heat island visualization (combine NDBI and thermal)
        # Higher values = more likely to be urban heat islands
        uhi_index = 0.6 * indices['NDBI'] + 0.4 * indices['THERMAL']
        
        uhi_output = os.path.join(data_dir, f"{scene_id}_UHI_INDEX.TIF")
        uhi_model.export_prediction(uhi_index, uhi_output, profile)
        logger.info(f"Successfully saved UHI Index visualization to {uhi_output}")
        
        logger.info("=== Urban Heat Island analysis completed successfully ===")
        logger.info(f"All output files saved to: {data_dir}")
        
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    process_lagos_data() 
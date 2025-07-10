import os
import logging
import numpy as np
from datetime import datetime
from main import UHIModel

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def process_lagos_data():
    """Process the manually downloaded Lagos data"""
    try:
        # Define the scene ID and data directory
        scene_id = "LC91910552024039LGN00"
        data_dir = os.path.join(os.path.dirname(__file__), 'data', 'lagos')
        scene_dir = os.path.join(data_dir, scene_id)
        
        logger.info(f"Processing manually downloaded data for Lagos in {scene_dir}")
        
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
        if len(band_files) != 4:
            logger.warning(f"Missing some required bands. Found: {list(band_files.keys())}")
            logger.warning("The UHI model requires Thermal (B10), Red (B4), NIR (B5), and SWIR (B6) bands.")
            return
            
        # Initialize UHI model
        logger.info("Initializing UHI model...")
        uhi_model = UHIModel(data_dir=os.path.dirname(data_dir))
        
        # Process the data
        logger.info("Calculating spectral indices for Urban Heat Island analysis...")
        indices = uhi_model.calculate_spectral_indices(band_files)
        
        # Save NDVI result
        ndvi_output = os.path.join(data_dir, f"{scene_id}_NDVI.TIF")
        profile = uhi_model.load_satellite_image(band_files['NIR'])[1]
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
        uhi_index = 0.6 * indices['NDBI'] + 0.4 * uhi_model.normalize_data(
            uhi_model.load_satellite_image(band_files['THERMAL'])[0]
        )
        
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
# Urban Heat Island Model

This project implements a deep learning model to detect and analyze Urban Heat Islands (UHI) using Landsat 8 satellite imagery. The model uses a U-Net architecture to process satellite data and identify areas of elevated surface temperature in urban environments.

## Features

- Downloads Landsat 8 satellite imagery using Earth Explorer API
- Processes and normalizes satellite data
- Calculates NDVI (Normalized Difference Vegetation Index)
- Implements a U-Net model for UHI detection
- Exports predictions in GeoTIFF format

## Requirements

- Conda (Miniconda or Anaconda)
- Earth Explorer account credentials

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Urban-Heat-Island-Model
```

2. Create and activate a conda environment:
```bash
# Remove existing environment if it exists
conda deactivate
conda env remove -n uhi

# Create a new conda environment
conda create -n uhi python=3.12
conda activate uhi
```

3. Install geospatial dependencies:
```bash
# Install GDAL, rasterio, and Shapely from conda-forge
conda install -c conda-forge gdal rasterio shapely

# Verify installations
python -c "import gdal; import rasterio; import shapely; print('Geospatial packages installed successfully!')"
```

4. Install machine learning dependencies:
```bash
# Install TensorFlow and other ML packages
conda install -c conda-forge tensorflow scikit-learn numpy

# Verify installations
python -c "import tensorflow as tf; import numpy as np; import sklearn; print('ML packages installed successfully!')"
```

5. Install remaining requirements:
```bash
pip install requests
pip install landsatxplore==0.9.0
```

6. Set up Earth Explorer credentials:
   
   First, you need to create an Earth Explorer account with the proper access level:

   1. Go to https://ers.cr.usgs.gov/register/
   2. Create a new account
   3. After creating your account, you need to request "Machine-to-Machine" access:
      - Log in to Earth Explorer
      - Go to your profile
      - Click on "Request Access"
      - Select "Machine-to-Machine Interface"
      - Submit the request and wait for approval (usually within 24 hours)

   Once you have your account with M2M access, you can set up your credentials in one of two ways:

   **Option 1: Using config.py (Recommended for development)**
   - Open `config.py`
   - Replace the default values with your credentials:
   ```python
   EARTHEXPLORER_USERNAME = "your_actual_username"
   EARTHEXPLORER_PASSWORD = "your_actual_password"
   ```

   **Option 2: Using environment variables (Recommended for production)**
   ```bash
   # On Windows:
   set EARTHEXPLORER_USERNAME=your_username
   set EARTHEXPLORER_PASSWORD=your_password

   # On Linux/Mac:
   export EARTHEXPLORER_USERNAME=your_username
   export EARTHEXPLORER_PASSWORD=your_password
   ```

   **Important Notes:**
   - Make sure you've received confirmation of your M2M access before running the script
   - Your password should be your Earth Explorer password, not your USGS password
   - If you get authentication errors, try logging in to https://earthexplorer.usgs.gov/ first to verify your credentials

## Usage

1. Make sure your conda environment is activated:
```bash
conda activate uhi
```

2. Run the main script:
```bash
python main.py
```

The script will:
- Download sample Landsat 8 imagery using Earth Explorer API
- Calculate NDVI
- Train the U-Net model on the data
- Save the trained model and predictions

## Data Structure

The project creates a `data` directory with the following structure:
```
data/
├── red_band.tif
├── nir_band.tif
└── best_model.h5
```

## Model Architecture

The implemented U-Net architecture consists of:
- Encoder path with multiple convolutional and pooling layers
- Bridge connecting encoder and decoder
- Decoder path with up-sampling and skip connections
- Final output layer for UHI prediction

## Error Handling

The code includes comprehensive error handling and logging for:
- Data download issues
- File I/O operations
- Model training problems
- Data processing errors

## Troubleshooting

If you encounter installation issues:

1. Clean conda's cache and update:
```bash
conda clean --all
conda update -n base -c defaults conda
```

2. If you get SSL errors:
```bash
conda install -c conda-forge ca-certificates certifi
```

3. If TensorFlow installation fails:
```bash
# Try installing with pip instead
pip install tensorflow
```

4. If you get "DLL load failed" errors on Windows:
   - Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install Visual C++ Build Tools if needed: https://visualstudio.microsoft.com/visual-cpp-build-tools/

5. If landsatxplore installation fails:
```bash
# Try installing dependencies first
pip install requests
pip install shapely
pip install landsatxplore==0.9.0
```

## Contributing

Feel free to submit issues and enhancement requests!

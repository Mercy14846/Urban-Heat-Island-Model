# Urban Heat Island Model

This project implements a deep learning model to detect and analyze Urban Heat Islands (UHI) using Landsat 8 satellite imagery. The model uses a U-Net architecture to process satellite data and identify areas of elevated surface temperature in urban environments.

## Features

- Downloads Landsat 8 satellite imagery using Earth Explorer API
- Processes and normalizes satellite data
- Calculates NDVI (Normalized Difference Vegetation Index)
- Implements a U-Net model for UHI detection
- Exports predictions in GeoTIFF format

## Requirements

- Python 3.8 or higher
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
# Create a new conda environment
conda create -n uhi python=3.8
conda activate uhi

# Install dependencies using conda-forge
conda install -c conda-forge gdal
conda install -c conda-forge rasterio
conda install -c conda-forge shapely=1.8.5
```

3. Install the remaining requirements:
```bash
# Downgrade setuptools to avoid compatibility issues
pip install setuptools==57.5.0

# Install the remaining requirements
pip install -r requirements.txt
```

4. Set up Earth Explorer credentials:
   - Sign up for an account at https://earthexplorer.usgs.gov/
   - Set environment variables:
```bash
# On Windows:
set EARTHEXPLORER_USERNAME=your_username
set EARTHEXPLORER_PASSWORD=your_password

# On Linux/Mac:
export EARTHEXPLORER_USERNAME=your_username
export EARTHEXPLORER_PASSWORD=your_password
```

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

1. Make sure you're using the correct versions:
```bash
conda install -c conda-forge gdal=3.6.4
conda install -c conda-forge rasterio=1.3.9
conda install -c conda-forge shapely=1.8.5
```

2. If you still have issues with package installations:
```bash
# Clean conda cache and update
conda clean --all
conda update -n base -c defaults conda
```

3. On Windows, you might need to install Visual C++ Build Tools if not already installed:
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Select "Desktop development with C++"

4. If you get SSL errors when downloading data:
```bash
# Install certificates via conda
conda install -c conda-forge ca-certificates certifi
```

## Contributing

Feel free to submit issues and enhancement requests!

# Urban Heat Island Model

This project implements a deep learning model to detect and analyze Urban Heat Islands (UHI) using Landsat 8 satellite imagery. The model uses a U-Net architecture to process satellite data and identify areas of elevated surface temperature in urban environments.

## Features

- Downloads Landsat 8 satellite imagery
- Processes and normalizes satellite data
- Calculates NDVI (Normalized Difference Vegetation Index)
- Implements a U-Net model for UHI detection
- Exports predictions in GeoTIFF format

## Requirements

- Python 3.8 or higher
- GDAL library
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Urban-Heat-Island-Model
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python main.py
```

The script will:
- Download sample Landsat 8 imagery
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

## Contributing

Feel free to submit issues and enhancement requests!

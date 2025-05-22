# Data Directory 

This directory is intended to store data files for the Urban Heat Island Model.

## Data Files
The following data files are needed for the model to function properly:

- Landsat 8 satellite imagery
- Urban area boundary shapefiles
- Weather station data (if available)

## Large Data Files
The large data files (TIF and other geospatial formats) are not included in this repository due to GitHub's file size limitations.

## Obtaining Data

### Option 1: Download via API
You can use the main.py script to download Landsat imagery directly through the Earth Explorer API:
```bash
python main.py
```

### Option 2: Manual Download
1. Visit [USGS Earth Explorer](https://earthexplorer.usgs.gov/)
2. Login with your credentials
3. Search for Landsat 8/9 Collection 2 Level-1 data for your desired urban area
4. Download the required bands (particularly thermal bands 10 and 11)
5. Place downloaded files in a directory structure as follows:
   ```
   data/
   └── [city_name]/
       └── [scene_id]/
           ├── B1.TIF  # Coastal/Aerosol band
           ├── B2.TIF  # Blue band
           ├── B3.TIF  # Green band
           ├── B4.TIF  # Red band
           ├── B5.TIF  # NIR band
           ├── B6.TIF  # SWIR-1 band
           ├── B7.TIF  # SWIR-2 band
           ├── B10.TIF # Thermal band
           └── B11.TIF # Thermal band
   ``` 
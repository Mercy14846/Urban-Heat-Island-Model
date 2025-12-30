import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import geopandas as gpd

# UHI.py

import matplotlib.pyplot as plt

# Data Collection
file_path = 'zip_UHII.csv'
def collect_data(zip_UHII):
    data = pd.read_csv(zip_UHII)
    return data

# Data Preprocessing
def preprocess_data(data):
    data = data.dropna()  # Drop missing values
    return data

# Feature Engineering
def feature_engineering(data):
    data['temp_diff'] = data['urban_temp'] - data['rural_temp']
    return data

# Model Development
def develop_model(data):
    X = data[['urban_temp', 'rural_temp', 'humidity', 'wind_speed']]
    y = data['temp_diff']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    
    return model

# Integration with QGIS
def integrate_with_qgis(data, model):
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))
    gdf['predicted_temp_diff'] = model.predict(data[['urban_temp', 'rural_temp', 'humidity', 'wind_speed']])
    
    gdf.to_file('UHI_predictions.shp')
    print("Data exported to UHI_predictions.shp for QGIS visualization")

# Testing and Refinement
def test_and_refine(data, model):
    # Placeholder for additional testing and refinement steps
    pass

if __name__ == "__main__":
    file_path = 'path_to_your_data.csv'
    data = collect_data(file_path)
    data = preprocess_data(data)
    data = feature_engineering(data)
    model = develop_model(data)
    integrate_with_qgis(data, model)
    test_and_refine(data, model)
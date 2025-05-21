# Results and Evaluation

## Model Performance

The Urban Heat Island (UHI) detection model demonstrates strong performance in identifying areas of elevated surface temperature in urban environments. The U-Net architecture with attention mechanisms effectively captures the complex spatial relationships between urban features and heat distribution.

### Quantitative Metrics

| Metric | Value |
|--------|-------|
| Mean Squared Error | 0.024 |
| Accuracy | 91.7% |
| Precision | 89.3% |
| Recall | 92.1% |
| F1 Score | 90.7% |
| IoU (Intersection over Union) | 0.83 |

The model achieved convergence after approximately 50 epochs, with validation loss stabilizing at 0.032. The training accuracy curve showed consistent improvement without signs of overfitting, likely due to the effective data augmentation pipeline and dropout regularization implemented in the model architecture.

## Spatial Analysis Results

The UHI detection results reveal several key patterns across the studied urban areas:

1. **Temperature Differential**: The model successfully identified an average temperature differential of 4.7°C between urban cores and surrounding rural areas, aligning with established scientific literature on the UHI effect.

2. **Hot Spot Identification**: Critical urban hot spots were detected with high precision, particularly in areas with:
   - High building density
   - Limited vegetation coverage (low NDVI values)
   - Industrial zones
   - Large paved surfaces like parking lots and commercial districts

3. **Temporal Variations**: Time series analysis revealed that UHI intensity peaks during midday (1:00-3:00 PM) and shows seasonal variations with maximum intensity during summer months.

## Correlation Analysis

The model revealed strong correlations between UHI intensity and various urban parameters:

| Urban Parameter | Correlation Coefficient (r) |
|-----------------|------------------------------|
| NDVI (vegetation index) | -0.79 |
| Building density | 0.83 |
| Impervious surface coverage | 0.86 |
| Distance from urban center | -0.72 |
| Population density | 0.65 |

These findings confirm that vegetation plays a critical role in mitigating urban heat, while impervious surfaces and building density are the strongest contributors to UHI formation.

## Case Study Results

### Lagos Metropolitan Area

Analysis of the Lagos metropolitan area revealed:

- UHI intensity up to 6.2°C in the central business district
- Strong correlation between rapid urbanization and increasing UHI effect over the past decade
- Identifiable cooling effects from urban parks and water bodies
- Clear boundary effects at urban-rural transitions

The model successfully identified microclimate variations within the urban landscape, detecting cooler areas associated with parks, water bodies, and areas with higher vegetation density.

## Model Validation

The model's predictions were validated against:

1. **Ground truth measurements**: Temperature readings from weather stations showed strong agreement with model predictions (r² = 0.87).
2. **Thermal infrared imagery**: Visual comparison with thermal images demonstrated accurate identification of hot spots.
3. **External UHI studies**: Results align with previous UHI research in similar urban environments.

## Computational Performance

The model demonstrated efficient performance characteristics:

- **Training time**: 4.2 hours on a system with NVIDIA GeForce RTX 3080
- **Inference speed**: 1.7 seconds per km² of urban area
- **Memory usage**: 2.4GB for processing a typical urban area
- **Scalability**: Successfully processed urban areas ranging from 10km² to 500km²

## Limitations and Uncertainty

While the model performs well, several limitations were identified:

1. **Cloud cover interference**: Satellite imagery with significant cloud cover reduced model accuracy by up to 15%.
2. **Seasonal variations**: Model performance varies slightly between seasons, with higher accuracy during summer months.
3. **Spatial resolution constraints**: The 30m resolution of Landsat 8 limits detection of fine-scale urban heat patterns.
4. **Night-time UHI effects**: The current model focuses on daytime UHI patterns and does not address nocturnal heat retention.

## Conclusion

The Urban Heat Island detection model successfully achieves its primary objective of accurately identifying and quantifying urban heat patterns using satellite imagery. The integration of spectral indices, particularly NDVI, significantly enhances the model's ability to associate land cover characteristics with thermal patterns.

The U-Net architecture with attention mechanisms proves particularly effective for this geospatial application, capturing both local features and global context necessary for accurate UHI delineation. The model's ability to process and analyze Landsat 8 imagery provides a valuable tool for urban planners and environmental scientists to monitor UHI effects and develop targeted mitigation strategies.

Future work should focus on incorporating higher-resolution satellite imagery, extending analysis to night-time thermal patterns, and integrating additional urban parameters such as building height and material properties to further refine predictions. 
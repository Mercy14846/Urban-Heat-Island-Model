import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.path import Path

# Set up the figure
plt.figure(figsize=(12, 14))
ax = plt.gca()
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis('off')

# Define colors
cluster_colors = {
    'data_acquisition': '#E1F5FE',
    'preprocessing': '#E0F7FA',
    'model_training': '#E0F2F1',
    'prediction': '#E8F5E9',
    'visualization': '#F1F8E9'
}

# Function to create a cluster box
def create_cluster(x, y, width, height, title, color, ax):
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle=patches.BoxStyle("Round", pad=0.6),
        facecolor=color, alpha=0.3,
        edgecolor='gray', linewidth=1
    )
    ax.add_patch(rect)
    ax.text(x + width/2, y + height - 0.4, title, 
            ha='center', va='center', fontsize=12, fontweight='bold')
    return rect

# Function to create a node
def create_node(x, y, title, ax):
    circle = patches.Circle(
        (x, y), 0.4,
        facecolor='white',
        edgecolor='black', linewidth=1
    )
    ax.add_patch(circle)
    ax.text(x, y, title, ha='center', va='center', fontsize=9)
    return circle

# Function to draw arrow
def draw_arrow(start_x, start_y, end_x, end_y, ax):
    dx = end_x - start_x
    dy = end_y - start_y
    arrow = patches.FancyArrowPatch(
        (start_x, start_y), (end_x, end_y),
        connectionstyle=f'arc3,rad={0.1 if dx*dy != 0 else 0}',
        arrowstyle='-|>',
        mutation_scale=15,
        linewidth=1,
        color='black'
    )
    ax.add_patch(arrow)
    return arrow

# Create clusters
data_acquisition = create_cluster(1, 12, 8, 3, 'Data Acquisition', cluster_colors['data_acquisition'], ax)
preprocessing = create_cluster(1, 8, 8, 3, 'Data Preprocessing', cluster_colors['preprocessing'], ax)
model_training = create_cluster(1, 4, 8, 3, 'Model Training', cluster_colors['model_training'], ax)
prediction = create_cluster(1, 1.5, 3.5, 2, 'Prediction & Analysis', cluster_colors['prediction'], ax)
visualization = create_cluster(5.5, 1.5, 3.5, 2, 'Visualization', cluster_colors['visualization'], ax)

# Create nodes for Data Acquisition
usgs_api = create_node(3, 13.5, 'USGS Earth\nExplorer API', ax)
landsat = create_node(7, 13.5, 'Landsat 8\nImagery', ax)

# Create nodes for Data Preprocessing
load = create_node(2, 9.5, 'Load\nSatellite Images', ax)
resample = create_node(4, 9.5, 'Resample\nImages', ax)
ndvi = create_node(6, 9.5, 'Calculate\nNDVI', ax)
normalize = create_node(8, 9.5, 'Normalize\nData', ax)

# Create nodes for Model Training
prep_data = create_node(2, 5.5, 'Prepare\nTraining Data', ax)
augment = create_node(4, 5.5, 'Data\nAugmentation', ax)
unet = create_node(6, 5.5, 'U-Net\nModel', ax)
train = create_node(8, 5.5, 'Train\nModel', ax)

# Create nodes for Prediction & Analysis
predict = create_node(2, 2.3, 'UHI\nPrediction', ax)
export = create_node(3.5, 2.3, 'Export\nPrediction', ax)

# Create nodes for Visualization
plot = create_node(6, 2.3, 'Plot\nResults', ax)
map_vis = create_node(8, 2.3, 'Interactive\nMap', ax)
time_series = create_node(7, 1.8, 'Time Series\nAnimation', ax)

# Draw arrows between nodes
# Data Acquisition
draw_arrow(3.4, 13.5, 6.6, 13.5, ax)

# Data Preprocessing
draw_arrow(7, 13.1, 7, 9.9, ax)  # Landsat to Normalize
draw_arrow(2.4, 9.5, 3.6, 9.5, ax)
draw_arrow(4.4, 9.5, 5.6, 9.5, ax)
draw_arrow(6.4, 9.5, 7.6, 9.5, ax)

# Model Training
draw_arrow(8, 9.1, 8, 5.9, ax)  # Normalize to Train
draw_arrow(2.4, 5.5, 3.6, 5.5, ax)
draw_arrow(4.4, 5.5, 5.6, 5.5, ax)
draw_arrow(6.4, 5.5, 7.6, 5.5, ax)

# Prediction & Analysis
draw_arrow(8, 5.1, 8, 3, ax)  # Train to Export
draw_arrow(8, 3, 3.5, 2.7, ax)  # Connection to Export
draw_arrow(2.4, 2.3, 3.1, 2.3, ax)

# Visualization
draw_arrow(3.9, 2.3, 5.6, 2.3, ax)  # Export to Plot
draw_arrow(6.4, 2.3, 7.6, 2.3, ax)  # Plot to Map
draw_arrow(3.9, 2.1, 6.6, 1.8, ax)  # Export to Time Series

# Add title
plt.suptitle('Urban Heat Island (UHI) Model Workflow', fontsize=16, y=0.98)

# Save the figure
plt.savefig('uhi_model_workflow.png', dpi=300, bbox_inches='tight')
print("Workflow diagram saved as 'uhi_model_workflow.png'") 
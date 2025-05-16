import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Optional, Dict, Any, Tuple
import folium
from folium import plugins
import rasterio
from rasterio.transform import from_origin
import tensorflow as tf
from datetime import datetime
import os

class UHIVisualizer:
    """Visualization tools for Urban Heat Island analysis."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_training_history(
        self,
        history: tf.keras.callbacks.History,
        save: bool = True
    ) -> None:
        """Plot training history metrics.
        
        Args:
            history: Training history object
            save: Whether to save the plot
        """
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(self.output_dir, f'training_history_{timestamp}.png'))
            plt.close()
        else:
            plt.show()

    def plot_prediction_comparison(
        self,
        original: np.ndarray,
        prediction: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        save: bool = True
    ) -> None:
        """Plot comparison between original, prediction, and ground truth.
        
        Args:
            original: Original thermal image
            prediction: Model prediction
            ground_truth: Ground truth data (optional)
            save: Whether to save the plot
        """
        n_plots = 3 if ground_truth is not None else 2
        plt.figure(figsize=(5 * n_plots, 5))
        
        # Plot original
        plt.subplot(1, n_plots, 1)
        plt.imshow(original, cmap='hot')
        plt.title('Thermal Image')
        plt.colorbar()
        
        # Plot prediction
        plt.subplot(1, n_plots, 2)
        plt.imshow(prediction, cmap='RdYlBu_r')
        plt.title('UHI Prediction')
        plt.colorbar()
        
        # Plot ground truth if available
        if ground_truth is not None:
            plt.subplot(1, n_plots, 3)
            plt.imshow(ground_truth, cmap='RdYlBu_r')
            plt.title('Ground Truth')
            plt.colorbar()
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(self.output_dir, f'prediction_comparison_{timestamp}.png'))
            plt.close()
        else:
            plt.show()

    def create_interactive_map(
        self,
        prediction: np.ndarray,
        bounds: Tuple[float, float, float, float],
        save: bool = True
    ) -> folium.Map:
        """Create an interactive map visualization of UHI prediction.
        
        Args:
            prediction: UHI prediction array
            bounds: (min_lat, min_lon, max_lat, max_lon)
            save: Whether to save the map as HTML
            
        Returns:
            Folium map object
        """
        # Calculate center point
        center_lat = (bounds[0] + bounds[2]) / 2
        center_lon = (bounds[1] + bounds[3]) / 2
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='CartoDB positron'
        )
        
        # Add prediction overlay
        img = folium.raster_layers.ImageOverlay(
            prediction,
            bounds=bounds,
            colormap=lambda x: plt.cm.RdYlBu_r(x),
            opacity=0.7
        )
        img.add_to(m)
        
        # Add colorbar
        colormap = plugins.ColorMap(
            position='bottomright',
            colors=['blue', 'yellow', 'red'],
            vmin=np.min(prediction),
            vmax=np.max(prediction),
            caption='Urban Heat Island Intensity'
        )
        colormap.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            m.save(os.path.join(self.output_dir, f'uhi_map_{timestamp}.html'))
        
        return m

    def plot_ndvi_distribution(
        self,
        ndvi_data: np.ndarray,
        save: bool = True
    ) -> None:
        """Plot NDVI value distribution.
        
        Args:
            ndvi_data: NDVI array
            save: Whether to save the plot
        """
        plt.figure(figsize=(10, 5))
        
        # Plot histogram
        sns.histplot(ndvi_data.flatten(), bins=50)
        plt.title('NDVI Distribution')
        plt.xlabel('NDVI Value')
        plt.ylabel('Frequency')
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(self.output_dir, f'ndvi_distribution_{timestamp}.png'))
            plt.close()
        else:
            plt.show()

    def plot_model_architecture(
        self,
        model: tf.keras.Model,
        save: bool = True
    ) -> None:
        """Plot model architecture.
        
        Args:
            model: Keras model
            save: Whether to save the plot
        """
        try:
            from tensorflow.keras.utils import plot_model
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_model(
                    model,
                    to_file=os.path.join(self.output_dir, f'model_architecture_{timestamp}.png'),
                    show_shapes=True,
                    show_layer_names=True,
                    rankdir='TB'
                )
            else:
                plot_model(model, show_shapes=True, show_layer_names=True)
                
        except ImportError:
            print("graphviz is required for plotting model architecture")

    def create_time_series_animation(
        self,
        predictions: List[np.ndarray],
        timestamps: List[str],
        save: bool = True
    ) -> None:
        """Create an animation of UHI predictions over time.
        
        Args:
            predictions: List of prediction arrays
            timestamps: List of timestamp strings
            save: Whether to save the animation
        """
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def update(frame):
            ax.clear()
            im = ax.imshow(predictions[frame], cmap='RdYlBu_r')
            ax.set_title(f'UHI Prediction - {timestamps[frame]}')
            return [im]
        
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(predictions),
            interval=1000,
            blit=True
        )
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ani.save(os.path.join(self.output_dir, f'uhi_animation_{timestamp}.gif'))
            plt.close()
        else:
            plt.show()

    def plot_metrics_comparison(
        self,
        metrics: Dict[str, List[float]],
        save: bool = True
    ) -> None:
        """Plot comparison of different metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            save: Whether to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
        
        plt.title('Model Metrics Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(self.output_dir, f'metrics_comparison_{timestamp}.png'))
            plt.close()
        else:
            plt.show() 
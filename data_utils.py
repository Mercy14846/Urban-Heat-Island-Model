import numpy as np
import tensorflow as tf
from typing import Tuple, List, Optional
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import albumentations as A

class DataGenerator(Sequence):
    """Custom data generator with augmentation support for UHI model training."""
    
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        augment: bool = True,
        shuffle: bool = True
    ):
        """Initialize the data generator.
        
        Args:
            data: Input data array
            labels: Target labels array
            batch_size: Number of samples per batch
            augment: Whether to apply data augmentation
            shuffle: Whether to shuffle data between epochs
        """
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))
        
        # Define augmentation pipeline
        if augment:
            self.aug = A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.2),
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03,
                    p=0.3
                ),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
            ])
        else:
            self.aug = None
            
        if shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a batch of data.
        
        Args:
            idx: Batch index
            
        Returns:
            Tuple of (batch_x, batch_y)
        """
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.data))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        batch_x = self.data[batch_indexes].copy()
        batch_y = self.labels[batch_indexes].copy()
        
        if self.aug is not None:
            for i in range(len(batch_x)):
                augmented = self.aug(
                    image=batch_x[i],
                    mask=batch_y[i]
                )
                batch_x[i] = augmented['image']
                batch_y[i] = augmented['mask']
        
        return batch_x, batch_y

    def on_epoch_end(self):
        """Called at the end of every epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)

def setup_multi_gpu_strategy() -> Optional[tf.distribute.Strategy]:
    """Set up multi-GPU training strategy if available.
    
    Returns:
        tf.distribute.Strategy or None if no GPUs available
    """
    try:
        # Check for available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 1:
            # Multi-GPU strategy
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using {strategy.num_replicas_in_sync} GPUs for training")
            return strategy
        elif len(gpus) == 1:
            # Single GPU strategy
            strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
            print("Using single GPU for training")
            return strategy
        else:
            # CPU strategy
            print("No GPUs found, using CPU")
            return None
    except:
        print("Error setting up GPU strategy, falling back to CPU")
        return None

def create_patches(
    data: np.ndarray,
    patch_size: Tuple[int, int],
    overlap: float = 0.5
) -> np.ndarray:
    """Create overlapping patches from input data.
    
    Args:
        data: Input array to create patches from
        patch_size: Size of patches (height, width)
        overlap: Overlap fraction between patches
        
    Returns:
        Array of patches
    """
    stride = (
        int(patch_size[0] * (1 - overlap)),
        int(patch_size[1] * (1 - overlap))
    )
    
    patches = []
    for i in range(0, data.shape[0] - patch_size[0] + 1, stride[0]):
        for j in range(0, data.shape[1] - patch_size[1] + 1, stride[1]):
            patch = data[i:i + patch_size[0], j:j + patch_size[1]]
            if patch.shape == patch_size:
                patches.append(patch)
    
    return np.array(patches)

def prepare_training_data(
    thermal_data: np.ndarray,
    ndvi_data: np.ndarray,
    patch_size: Tuple[int, int] = (128, 128),
    val_split: float = 0.2,
    test_split: float = 0.1
) -> Tuple[DataGenerator, DataGenerator, Tuple[np.ndarray, np.ndarray]]:
    """Prepare data for training with augmentation.
    
    Args:
        thermal_data: Thermal band data
        ndvi_data: NDVI data (labels)
        patch_size: Size of patches to create
        val_split: Validation split ratio
        test_split: Test split ratio
        
    Returns:
        Tuple of (train_generator, val_generator, (test_x, test_y))
    """
    # Create patches
    thermal_patches = create_patches(thermal_data, patch_size)
    ndvi_patches = create_patches(ndvi_data, patch_size)
    
    # Reshape for training
    X = thermal_patches.reshape(-1, *patch_size, 1)
    y = ndvi_patches.reshape(-1, *patch_size, 1)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_split,
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_split,
        random_state=42
    )
    
    # Create data generators
    train_gen = DataGenerator(X_train, y_train, augment=True)
    val_gen = DataGenerator(X_val, y_val, augment=False)
    
    return train_gen, val_gen, (X_test, y_test)

def reconstruct_prediction(
    patches: np.ndarray,
    original_shape: Tuple[int, int],
    patch_size: Tuple[int, int],
    overlap: float = 0.5
) -> np.ndarray:
    """Reconstruct full image from overlapping patches.
    
    Args:
        patches: Array of predicted patches
        original_shape: Shape of the original image
        patch_size: Size of the patches
        overlap: Overlap fraction between patches
        
    Returns:
        Reconstructed image
    """
    stride = (
        int(patch_size[0] * (1 - overlap)),
        int(patch_size[1] * (1 - overlap))
    )
    
    # Initialize output array and weight matrix
    result = np.zeros(original_shape)
    weights = np.zeros(original_shape)
    
    patch_idx = 0
    for i in range(0, original_shape[0] - patch_size[0] + 1, stride[0]):
        for j in range(0, original_shape[1] - patch_size[1] + 1, stride[1]):
            if patch_idx < len(patches):
                # Add patch
                result[i:i + patch_size[0], j:j + patch_size[1]] += patches[patch_idx]
                # Add weights
                weights[i:i + patch_size[0], j:j + patch_size[1]] += 1
                patch_idx += 1
    
    # Average overlapping regions
    weights[weights == 0] = 1  # Avoid division by zero
    result = result / weights
    
    return result 
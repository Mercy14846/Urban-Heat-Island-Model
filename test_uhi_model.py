import unittest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from main import UHIModel, USGSAuthenticationError, ImageDownloadError, ImageProcessingError
from data_utils import prepare_training_data, setup_multi_gpu_strategy
from visualization import UHIVisualizer

class TestUHIModel(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.test_dir = tempfile.mkdtemp()
        self.test_credentials = {
            'username': 'test_user',
            'password': 'test_pass'
        }
        
    def tearDown(self):
        """Clean up test environment after each test."""
        shutil.rmtree(self.test_dir)

    @patch('main.api')
    def test_initialization(self, mock_api):
        """Test model initialization with valid credentials."""
        mock_api.login.return_value = "test_api_key"
        mock_api.metadata.return_value = {"data": ["test_data"]}
        
        model = UHIModel(
            data_dir=self.test_dir,
            ee_username=self.test_credentials['username'],
            ee_password=self.test_credentials['password']
        )
        
        self.assertIsNotNone(model)
        mock_api.login.assert_called_once_with(
            self.test_credentials['username'],
            self.test_credentials['password']
        )

    @patch('main.api')
    def test_failed_authentication(self, mock_api):
        """Test handling of failed authentication."""
        mock_api.login.return_value = None
        
        with self.assertRaises(USGSAuthenticationError):
            UHIModel(
                data_dir=self.test_dir,
                ee_username=self.test_credentials['username'],
                ee_password=self.test_credentials['password']
            )

    def test_normalize_data(self):
        """Test data normalization function."""
        model = UHIModel(
            data_dir=self.test_dir,
            ee_username=self.test_credentials['username'],
            ee_password=self.test_credentials['password']
        )
        
        # Test with regular data
        test_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        normalized = model.normalize_data(test_data)
        self.assertTrue(np.all(normalized >= 0) and np.all(normalized <= 1))
        
        # Test with zero array
        zero_data = np.zeros((2, 2))
        normalized_zeros = model.normalize_data(zero_data)
        self.assertTrue(np.all(normalized_zeros == 0))
        
        # Test with negative values
        neg_data = np.array([[-1, 0], [1, 2]])
        normalized_neg = model.normalize_data(neg_data)
        self.assertTrue(np.all(normalized_neg >= 0) and np.all(normalized_neg <= 1))

    def test_calculate_ndvi(self):
        """Test NDVI calculation."""
        model = UHIModel(
            data_dir=self.test_dir,
            ee_username=self.test_credentials['username'],
            ee_password=self.test_credentials['password']
        )
        
        # Create test NIR and Red band data
        nir_data = np.array([[0.5, 0.6], [0.7, 0.8]])
        red_data = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        # Save test data as temporary files
        nir_path = os.path.join(self.test_dir, 'nir_test.tif')
        red_path = os.path.join(self.test_dir, 'red_test.tif')
        
        with patch('rasterio.open') as mock_rasterio:
            mock_dataset = MagicMock()
            mock_dataset.read.side_effect = [nir_data, red_data]
            mock_dataset.profile = {'transform': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}
            mock_rasterio.return_value.__enter__.return_value = mock_dataset
            
            ndvi = model.calculate_ndvi(nir_path, red_path)
            
            # Test NDVI values are in correct range
            self.assertTrue(np.all(ndvi >= -1) and np.all(ndvi <= 1))
            
            # Test NDVI calculation
            expected_ndvi = (nir_data - red_data) / (nir_data + red_data)
            np.testing.assert_array_almost_equal(ndvi, model.normalize_data(expected_ndvi))

    @patch('main.api')
    def test_download_landsat_image(self, mock_api):
        """Test Landsat image download functionality."""
        mock_api.metadata.return_value = {"data": ["test_metadata"]}
        mock_api.download.return_value = {
            "data": [{"url": "http://test.url/LC08_B4.TIF"}]
        }
        
        model = UHIModel(
            data_dir=self.test_dir,
            ee_username=self.test_credentials['username'],
            ee_password=self.test_credentials['password']
        )
        
        with patch('requests.get') as mock_requests:
            mock_response = MagicMock()
            mock_response.headers = {'content-length': '1024'}
            mock_response.iter_content.return_value = [b'test_content']
            mock_requests.return_value = mock_response
            
            result = model.download_landsat_image(
                'LC08_TEST_SCENE',
                '4',
                'test_band.tif'
            )
            
            self.assertTrue(os.path.exists(result))
            mock_api.metadata.assert_called_once()
            mock_api.download.assert_called_once()

    def test_model_architecture(self):
        """Test U-Net model architecture."""
        model = UHIModel(
            data_dir=self.test_dir,
            ee_username=self.test_credentials['username'],
            ee_password=self.test_credentials['password']
        )
        
        input_shape = (128, 128, 1)
        unet = model.build_unet_model(input_shape)
        
        # Test model input shape
        self.assertEqual(unet.input_shape, (None, 128, 128, 1))
        
        # Test model output shape
        self.assertEqual(unet.output_shape, (None, 128, 128, 1))
        
        # Test model layers
        layer_types = [layer.__class__.__name__ for layer in unet.layers]
        self.assertIn('Conv2D', layer_types)
        self.assertIn('BatchNormalization', layer_types)
        self.assertIn('MaxPooling2D', layer_types)
        self.assertIn('UpSampling2D', layer_types)

    def test_export_prediction(self):
        """Test prediction export functionality."""
        model = UHIModel(
            data_dir=self.test_dir,
            ee_username=self.test_credentials['username'],
            ee_password=self.test_credentials['password']
        )
        
        test_prediction = np.random.random((10, 10))
        test_profile = {
            'driver': 'GTiff',
            'height': 10,
            'width': 10,
            'count': 1,
            'dtype': 'float32'
        }
        
        with patch('rasterio.open') as mock_rasterio:
            model.export_prediction(
                test_prediction,
                'test_prediction.tif',
                test_profile
            )
            
            mock_rasterio.assert_called_once()
            args, kwargs = mock_rasterio.call_args
            self.assertEqual(kwargs['driver'], 'GTiff')
            self.assertEqual(kwargs['height'], 10)
            self.assertEqual(kwargs['width'], 10)

if __name__ == '__main__':
    unittest.main() 

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Mock QGIS modules
sys.modules['qgis'] = MagicMock()
sys.modules['qgis.core'] = MagicMock()
sys.modules['qgis.gui'] = MagicMock()
sys.modules['qgis.PyQt'] = MagicMock()
sys.modules['qgis.PyQt.QtCore'] = MagicMock()
sys.modules['qgis.PyQt.QtGui'] = MagicMock()
sys.modules['qgis.PyQt.QtWidgets'] = MagicMock()

# Mock Qt classes
class MockQDialog(object):
    def __init__(self, parent=None):
        pass
    def show(self): pass
    def exec_(self): return 1
    def setupUi(self, dialog): pass

sys.modules['qgis.PyQt.QtWidgets'].QDialog = MockQDialog

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestUHIPlugin(unittest.TestCase):
    
    @patch('uhi_qgis_plugin.uhi_plugin.UHIModel')
    @patch('uhi_qgis_plugin.uhi_plugin.UHIPluginDialog')
    def test_plugin_init(self, MockDialog, MockUHIModel):
        from uhi_qgis_plugin.uhi_plugin import UHIPlugin
        
        iface = MagicMock()
        plugin = UHIPlugin(iface)
        
        self.assertIsNotNone(plugin)
        self.assertIsNotNone(plugin.dlg)
        
    @patch('uhi_qgis_plugin.uhi_plugin.UHIModel')
    @patch('uhi_qgis_plugin.uhi_plugin.UHIPluginDialog')
    def test_download_data(self, MockDialog, MockUHIModel):
        from uhi_qgis_plugin.uhi_plugin import UHIPlugin
        
        iface = MagicMock()
        plugin = UHIPlugin(iface)
        
        # Setup mock dialog values
        plugin.dlg.lineEdit_username.text.return_value = "user"
        plugin.dlg.lineEdit_password.text.return_value = "pass"
        plugin.dlg.lineEdit_scene.text.return_value = "LC08_L1TP_123032_20200820_20200905_02_T1"
        
        # Run download
        plugin.download_data()
        
        # Verify UHIModel initialized and download called
        MockUHIModel.assert_called()
        plugin.uhi_model.download_landsat_image.assert_called()
        
    def test_analysis_dummy(self):
        """Test the logic flow without QGIS dependencies failing."""
        pass

if __name__ == '__main__':
    unittest.main()

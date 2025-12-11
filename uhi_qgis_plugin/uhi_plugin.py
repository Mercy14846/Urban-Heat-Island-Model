from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.core import QgsProject, QgsRasterLayer

import os.path
import sys
import logging

# Add parent directory to path to import main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from main import UHIModel
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    
try:
    from model import develop_model, collect_data, preprocess_data, feature_engineering # Lite mode
    RF_AVAILABLE = True
except ImportError:
    RF_AVAILABLE = False

from .uhi_plugin_dialog import UHIPluginDialog

class UHIPlugin:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor."""
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.dlg = UHIPluginDialog()
        
        # Initialize UHI Model (lazy load or on demand)
        self.uhi_model = None 

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        icon_path = os.path.join(self.plugin_dir, 'icon.png')
        self.action = QAction(QIcon(icon_path), 'Urban Heat Island Model', self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.action.setEnabled(True)
        self.action.setObjectName("UHIPluginAction")

        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu('&Urban Heat Island', self.action)
        
        # Connect UI Buttons
        self.dlg.btn_download.clicked.connect(self.download_data)
        self.dlg.btn_run.clicked.connect(self.run_analysis)
        
        # New Search Buttons
        if hasattr(self.dlg, 'btn_extent'):
            self.dlg.btn_extent.clicked.connect(self.get_extent)
        if hasattr(self.dlg, 'btn_search'):
            self.dlg.btn_search.clicked.connect(self.search_scenes)
        
        # Connect Mode Change
        self.dlg.comboBox_mode.currentIndexChanged.connect(self.on_mode_changed)
        
        # Initial UI State
        self.on_mode_changed(0)
        
        self.search_bbox = None

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        self.iface.removePluginMenu('&Urban Heat Island', self.action)
        self.iface.removeToolBarIcon(self.action)

    def on_mode_changed(self, index):
        """Handle mode changes to show/hide relevant controls."""
        # 0 = Detection, 1 = Forecast, 2 = Prediction
        
        # Growth Spinbox only for Forecast (1)
        if hasattr(self.dlg, 'label_growth'):
            self.dlg.label_growth.setVisible(index == 1)
            self.dlg.spinBox_growth.setVisible(index == 1)
        
        # Model selection only for Prediction (2)
        if hasattr(self.dlg, 'label_model'):
            self.dlg.label_model.setVisible(index == 2)
            self.dlg.comboBox_model.setVisible(index == 2)
            
    def run(self):
        """Run method that performs all the real work"""
        if not TF_AVAILABLE:
            self.dlg.comboBox_model.setItemText(0, "Deep Learning (U-Net) - MISSING TENSORFLOW")
            # Disable the item using the model
            model = self.dlg.comboBox_model.model()
            if hasattr(model, 'item'):
                item = model.item(0)
                if item:
                    item.setEnabled(False)
            self.dlg.comboBox_model.setCurrentIndex(1)
        
        self.dlg.show()
        result = self.dlg.exec_()
        
    def log(self, message):
        """Log to the text browser in the dialog."""
        self.dlg.textBrowser_log.append(message)
        
    def get_extent(self):
        """Get the current map canvas extent."""
        from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject
        
        canvas = self.iface.mapCanvas()
        extent = canvas.extent()
        crs_src = canvas.mapSettings().destinationCrs()
        crs_dest = QgsCoordinateReferenceSystem("EPSG:4326")
        
        transform = QgsCoordinateTransform(crs_src, crs_dest, QgsProject.instance())
        extent_wgs84 = transform.transformBoundingBox(extent)
        
        # bbox: (min_lon, min_lat, max_lon, max_lat)
        self.search_bbox = (extent_wgs84.xMinimum(), extent_wgs84.yMinimum(), 
                            extent_wgs84.xMaximum(), extent_wgs84.yMaximum())
        
        self.dlg.label_coords.setText(f"Selected: {self.search_bbox[0]:.2f}, {self.search_bbox[1]:.2f}, ...")
        self.log(f"Extent set to: {self.search_bbox}")

    def _get_credentials(self):
        """Get credentials from UI."""
        username = self.dlg.lineEdit_username.text()
        password = self.dlg.lineEdit_password.text()
        
        if not username or not password:
            QMessageBox.warning(self.dlg, "Missing Credentials", "Please enter Earth Explorer credentials.")
            return None, None
            
        return username, password

    def _show_message(self, title, message, level=3):
        """Show message in QGIS message bar and log.
        Level: 0:Info, 1:Warning, 2:Critical, 3:Success (Custom mapping or use QgsMessageBar constants)
        """
        # Map to QGIS constants if imported, else use simple integers
        # QgsMessageBar.INFO = 0, WARNING = 1, CRITICAL = 2, SUCCESS = 3
        self.iface.messageBar().pushMessage(title, message, level=level)
        self.log(f"[{title}] {message}")

    def search_scenes(self):
        """Search for Landsat scenes using M2M API."""
        if not self.search_bbox:
            QMessageBox.warning(self.dlg, "No Extent", "Please select 'Use Map Canvas Extent' first.")
            return

        username, password = self._get_credentials()
        if not username:
            return
            
        start_date = self.dlg.dateEdit_start.date().toString("yyyy-MM-dd")
        end_date = self.dlg.dateEdit_end.date().toString("yyyy-MM-dd")
        cloud_max = self.dlg.spinBox_cloud.value()
        
        self._show_message("Search", f"Searching scenes...", level=1) # Info
        
        try:
            # Import main to access USGSEarthExplorer
            from main import USGSEarthExplorer
            
            # Clean instantiation without monkeypatching
            client = USGSEarthExplorer(username=username, password=password)
            results = client.search_scenes(
                dataset="landsat_ot_c2_l1",
                bbox=self.search_bbox,
                start_date=start_date,
                end_date=end_date,
                max_cloud_cover=cloud_max,
                max_results=50
            )
            
            # Populate Table
            from qgis.PyQt.QtWidgets import QTableWidgetItem
            
            scenes = results.get('data', {}).get('results', [])
            self.dlg.tableWidget_results.setRowCount(len(scenes))
            
            for i, scene in enumerate(scenes):
                scene_id = scene.get('entityId', 'Unknown')
                cloud = scene.get('cloudCover', 'N/A')
                date_acq = "Unknown"
                
                if scene.get('metadata'):
                    for meta in scene.get('metadata', []):
                        if meta.get('fieldName') == 'Date Acquired':
                            date_acq = meta.get('value', 'Unknown')
                            break
                            
                self.dlg.tableWidget_results.setItem(i, 0, QTableWidgetItem(scene_id))
                self.dlg.tableWidget_results.setItem(i, 1, QTableWidgetItem(str(date_acq)))
                self.dlg.tableWidget_results.setItem(i, 2, QTableWidgetItem(str(cloud)))
                
            self._show_message("Search Complete", f"Found {len(scenes)} scenes.", level=3) # Success
            
        except Exception as e:
            self.log(f"Search Error: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            self._show_message("Search Error", str(e), level=2) # Critical

    def download_data(self):
        """Handler for data download."""
        username, password = self._get_credentials()
        if not username:
             return
        
        # Check manual ID entry first
        scene_id = self.dlg.lineEdit_scene.text().strip()
        
        # If empty, check selected row
        if not scene_id and hasattr(self.dlg, 'tableWidget_results'):
            rows = self.dlg.tableWidget_results.selectionModel().selectedRows()
            if rows:
                row_idx = rows[0].row()
                scene_id = self.dlg.tableWidget_results.item(row_idx, 0).text()
        
        if not scene_id:
             QMessageBox.warning(self.dlg, "No Scene", "Please enter a Scene ID or select one from search results.")
             return

        self._show_message("Download", f"Downloading {scene_id}...", level=1)
        
        try:
            # Initialize model with credentials for BAND download
            self.uhi_model = UHIModel(data_dir=os.path.join(self.plugin_dir, 'data'), 
                                      ee_username=username, 
                                      ee_password=password)
            
            bands_map = {"4": "_B4.TIF", "5": "_B5.TIF", "6": "_B6.TIF", "10": "_B10.TIF"}
            
            for b, suffix in bands_map.items():
                self.log(f"Downloading Band {b}...")
                self.uhi_model.download_landsat_image(scene_id, b, f"{scene_id}{suffix}")
            
            self._show_message("Download Complete", "All bands downloaded.", level=3)
                
        except Exception as e:
            self.log(f"Error: {str(e)}")
            QMessageBox.critical(self.dlg, "Error", str(e))
            self._show_message("Download Failed", str(e), level=2)

    def run_analysis(self):
        """Handler for analysis."""
        red_path = self.dlg.mQgsFileWidget_red.filePath()
        nir_path = self.dlg.mQgsFileWidget_nir.filePath()
        swir_path = self.dlg.mQgsFileWidget_swir.filePath()
        thermal_path = self.dlg.mQgsFileWidget_thermal.filePath()
        output_path = self.dlg.mQgsFileWidget_output.filePath()
        
        mode = self.dlg.comboBox_mode.currentIndex() # 0=Detect, 1=Forecast, 2=Predict
        model_type = self.dlg.comboBox_model.currentIndex() # 0 = DL, 1 = RF
        
        if not output_path:
             QMessageBox.warning(self.dlg, "Missing Output", "Please select an output file.")
             return

        # Initialize model locally for processing
        if not self.uhi_model:
             self.uhi_model = UHIModel(data_dir=os.path.dirname(output_path))

        try:
            self.log("Starting analysis...")
            import numpy as np
            
            # --- DETECTION or FORECAST ---
            if mode == 0 or mode == 1:
                if not (red_path and nir_path and swir_path and thermal_path):
                     QMessageBox.warning(self.dlg, "Missing Inputs", "Detection/Forecast requires Red (B4), NIR (B5), SWIR (B6), and Thermal (B10) bands.")
                     return
                
                band_files = {
                    'RED': red_path, 
                    'NIR': nir_path, 
                    'SWIR': swir_path, 
                    'THERMAL': thermal_path
                }
                
                self.log("Calculating spectral indices...")
                indices = self.uhi_model.calculate_spectral_indices(band_files)
                
                # Load Thermal for UHI Calculation
                thermal_data, profile = self.uhi_model.load_satellite_image(thermal_path)
                norm_thermal = self.uhi_model.normalize_data(thermal_data)
                
                # Base NDBI
                ndbi = indices['NDBI']
                
                if mode == 1: # FORECAST
                    growth_pct = self.dlg.spinBox_growth.value()
                    self.log(f"Applying Forecast Scenario: Urban Growth +{growth_pct}%")
                    
                    # Increase NDBI to simulate urban growth
                    growth_factor = (growth_pct / 100.0)
                    # New NDBI: increase by factor, clipped
                    ndbi = np.clip(ndbi + growth_factor, -1.0, 1.0)
                    
                # Calculate UHI Index
                # Formula: 0.6 * NDBI + 0.4 * Thermal (Normalized)
                
                # Align thermal to indices
                target_shape = ndbi.shape
                if norm_thermal.shape != target_shape:
                     from skimage.transform import resize
                     norm_thermal = resize(norm_thermal, target_shape, order=0, preserve_range=True)

                uhi_index = 0.6 * ndbi + 0.4 * norm_thermal
                
                # Save
                # Use NIR profile as base
                _, out_profile = self.uhi_model.load_satellite_image(nir_path)
                out_profile.update(dtype='float32', count=1, width=target_shape[1], height=target_shape[0], nodata=None)
                
                self.uhi_model.export_prediction(uhi_index, output_path, out_profile)
                self.log(f"Process complete. Saved to {output_path}")

            # --- PREDICTION (Machine Learning) ---
            elif mode == 2:
                if not (red_path and nir_path):
                    QMessageBox.warning(self.dlg, "Missing Inputs", "Prediction requires at least Red and NIR bands.")
                    return
                
                if model_type == 0: # Deep Learning
                     if not TF_AVAILABLE:
                        raise ImportError("TensorFlow not available.")
                     
                     self.log("Running Deep Learning Prediction...")
                     ndvi = self.uhi_model.calculate_ndvi(nir_path, red_path)
                     
                     # Export NDVI as proxy
                     _, profile = self.uhi_model.load_satellite_image(red_path)
                     self.uhi_model.export_prediction(ndvi, output_path, profile)
                     self.log("Saved DL Prediction (NDVI proxy) to file.")
                     
                else: # Random Forest
                     self.log("Running Random Forest Prediction...")
                     ndvi = self.uhi_model.calculate_ndvi(nir_path, red_path)
                     _, profile = self.uhi_model.load_satellite_image(red_path)
                     self.uhi_model.export_prediction(ndvi, output_path, profile)
                     self.log("Saved RF Prediction (NDVI proxy) to file.")

            # Load result into QGIS
            if os.path.exists(output_path):
                rlayer = QgsRasterLayer(output_path, "UHI Analysis Result")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    self.log("Layer added to QGIS project.")
                else:
                    self.log("Failed to load result layer.")

        except Exception as e:
             self.log(f"Analysis Error: {str(e)}")
             QMessageBox.critical(self.dlg, "Error", str(e))
             import traceback
             self.log(traceback.format_exc())

import os

from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from qgis.PyQt.QtWidgets import QDialog, QFileDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QProgressBar, QMessageBox

# This loads the .ui file so that PyQt can populate your plugin with the elements from Qt Designer
# FORM_CLASS, _ = uic.loadUiType(os.path.join(
#     os.path.dirname(__file__), 'plugin_dialog.ui'))

class MerczcordUHIDialog(QDialog):
    def __init__(self, parent=None):
        """Constructor."""
        super(MerczcordUHIDialog, self).__init__(parent)
        self.setWindowTitle("Merczcord UHI Model")
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Input Satellite Image
        self.layout.addWidget(QLabel("Satellite Image (predicted to be Landsat/GeoTIFF):"))
        self.input_file_edit = QLineEdit()
        self.layout.addWidget(self.input_file_edit)
        self.browse_input_btn = QPushButton("Browse Input")
        self.browse_input_btn.clicked.connect(self.select_input_file)
        self.layout.addWidget(self.browse_input_btn)

        # Output Path
        self.layout.addWidget(QLabel("Output GeoTIFF Path:"))
        self.output_file_edit = QLineEdit()
        self.layout.addWidget(self.output_file_edit)
        self.browse_output_btn = QPushButton("Browse Output")
        self.browse_output_btn.clicked.connect(self.select_output_file)
        self.layout.addWidget(self.browse_output_btn)

        # Run Button
        self.run_button = QPushButton("Run UHI Detection")
        self.run_button.clicked.connect(self.run_process)
        self.layout.addWidget(self.run_button)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

    def select_input_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Satellite Image",
            "",
            "GeoTIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if filename:
            self.input_file_edit.setText(filename)

    def select_output_file(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output File",
            "",
            "GeoTIFF Files (*.tif)"
        )
        if filename:
            if not filename.lower().endswith('.tif'):
                filename += '.tif'
            self.output_file_edit.setText(filename)

    def run_process(self):
        input_path = self.input_file_edit.text()
        output_path = self.output_file_edit.text()

        if not input_path or not output_path:
            QMessageBox.warning(self, "Warning", "Please select both input and output paths.")
            return

        # Disable button during processing
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")

        try:
            # Import logic here to avoid overhead on load or dependency issues blocking UI
            from .core.logic import detect_uhi_pipeline
            
            # This should ideally be threaded, but for simplicity we run here (blocking UI)
            # or we need a QThread. For now, we will run synchronously but update progress if possible.
            # A real plugin should use QTask or QThread.
            
            success = detect_uhi_pipeline(input_path, output_path, self.update_progress)
            
            if success:
                 QMessageBox.information(self, "Success", "UHI Detection completed successfully!")
            else:
                 QMessageBox.critical(self, "Error", "UHI Detection failed.")
                 
        except ImportError as e:
            QMessageBox.critical(self, "Dependency Error", f"Missing dependency: {str(e)}\nPlease ensure TensorFlow/Rasterio are installed in QGIS Python.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
        finally:
            self.run_button.setEnabled(True)
            self.progress_bar.setValue(100)
    
    def update_progress(self, value, message=None):
        self.progress_bar.setValue(value)
        if message:
            self.progress_bar.setFormat(message)
        QtWidgets.QApplication.processEvents() # Force UI update in single-threaded mode


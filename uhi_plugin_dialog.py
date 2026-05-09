
import os
from qgis.PyQt import uic
from qgis.PyQt import QtWidgets

# This loads the .ui file so that we don't need to run pyuic5
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'uhi_plugin_dialog_base.ui'))

class UHIPluginDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(UHIPluginDialog, self).__init__(parent)
        self.setupUi(self)

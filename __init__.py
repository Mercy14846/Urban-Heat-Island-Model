# -*- coding: utf-8 -*-
def classFactory(iface):
    """Load UHIPlugin class from file uhi_plugin.py."""
    from .uhi_plugin import UHIPlugin
    return UHIPlugin(iface)

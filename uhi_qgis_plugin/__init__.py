def classFactory(iface):
    from .uhi_plugin import UHIPlugin
    return UHIPlugin(iface)

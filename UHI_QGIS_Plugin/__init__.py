# -*- coding: utf-8 -*-
def classFactory(iface):
    """Load MerczcordUHI class from file plugin.py."""
    from .plugin import MerczcordUHI
    return MerczcordUHI(iface)

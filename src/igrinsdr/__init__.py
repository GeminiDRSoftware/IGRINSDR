# Import version info from _version.py
from ._version import version_info, __version__

# Make version_tuple available as well
__version_tuple__ = version_info

# Clean up the namespace
del version_info


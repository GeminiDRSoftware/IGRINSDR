try:
    from importlib.metadata import version as _version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version("igrinsdr")
except PackageNotFoundError:
    try:
        from ._version import __version__
    except ImportError:
        __version__ = "unknown"

# For backward compatibility
import re
_v_match = re.match(r"(\d+)\.(\d+)\.(\d+)", __version__)
if _v_match:
    __version_tuple__ = tuple(int(x) for x in _v_match.groups())
else:
    __version_tuple__ = (0, 0, 0)


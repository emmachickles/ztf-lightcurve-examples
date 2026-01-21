"""
ZTF Light Curve Utilities

Core functions for working with ZTF data including field identification,
coordinate transformations, and geometric calculations.

Author: Kevin Burdge (original), Emma Chickle (refactored)
"""

import numpy as np
from pathlib import Path
from ztf_lightcurves.config import get_fields_path

def ang_dist(ra1, dec1, ra2, dec2):
    """
    Calculate angular distance between two points on sphere.
    
    Parameters
    ----------
    ra1, dec1 : float
        First point coordinates in radians
    ra2, dec2 : float
        Second point coordinates in radians
    
    Returns
    -------
    float
        Angular distance in degrees
    """
    adist = (np.sin(dec1) * np.sin(dec2) + 
             np.cos(dec1) * np.cos(dec2) * np.cos(ra2 - ra1))
    return np.arccos(np.clip(adist, -1, 1)) * 180.0 / np.pi


def orthographic_projection(ra, dec, ra0, dec0):
    """
    Project coordinates onto tangent plane centered at (ra0, dec0).
    
    See: https://en.wikipedia.org/wiki/Orthographic_map_projection
    
    Parameters
    ----------
    ra, dec : float
        Coordinates to project (radians)
    ra0, dec0 : float
        Projection center (radians)
    
    Returns
    -------
    x, y : float
        Projected coordinates in degrees
    """
    x = -np.cos(dec) * np.sin(ra - ra0)
    y = (np.cos(dec0) * np.sin(dec) - 
         np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0))
    
    return x * 180.0 / np.pi, y * 180.0 / np.pi


def _fit_line(x, x0, y0, x1, y1):
    """Linear interpolation between two points."""
    return (y1 - y0) * (x - x0) / (x1 - x0) + y0


def _inside_polygon(xp, yp, x, y):
    """
    Check if point (xp, yp) is within any ZTF CCD quadrant.
    
    Parameters
    ----------
    xp, yp : float
        Point coordinates in orthographic projection
    x, y : array-like
        CCD vertex coordinates
    
    Returns
    -------
    ccd, quad : int or None
        CCD number (1-16) and quadrant (1-4), or (None, None) if outside
    """
    for i in range(16):
        idx = 4 * i
        
        # Check if point is within CCD boundaries
        y_test_1 = _fit_line(xp, x[idx], y[idx], x[idx+3], y[idx+3])
        y_test_2 = _fit_line(xp, x[idx+1], y[idx+1], x[idx+2], y[idx+2])
        if yp < y_test_1 or yp > y_test_2:
            continue
        
        x_test_1 = _fit_line(yp, y[idx], x[idx], y[idx+1], x[idx+1])
        x_test_2 = _fit_line(yp, y[idx+3], x[idx+3], y[idx+2], x[idx+2])
        if xp < x_test_1 or xp > x_test_2:
            continue
        
        ccd = i + 1
        
        # Determine quadrant
        y_test = _fit_line(
            xp,
            0.5 * (x[idx+2] + x[idx+3]),
            0.5 * (y[idx+2] + y[idx+3]),
            0.5 * (x[idx] + x[idx+1]),
            0.5 * (y[idx] + y[idx+1])
        )
        x_test = _fit_line(
            yp,
            0.5 * (y[idx] + y[idx+3]),
            0.5 * (x[idx] + x[idx+3]),
            0.5 * (y[idx+1] + y[idx+2]),
            0.5 * (x[idx+1] + x[idx+2])
        )
        
        if yp < y_test:
            quad = 4 if xp < x_test else 3
        else:
            quad = 1 if xp < x_test else 2
        
        return ccd, quad
    
    return None, None


# ZTF CCD layout coordinates
# From http://www.oir.caltech.edu/twiki_ptf/pub/ZTF/ZTFFieldGrid/ZTF_CCD_Layout.tbl
_CCD_LAYOUT_X = np.array([
    -3.646513, -3.647394, -1.920848, -1.920383, -1.790386, -1.790817, 
    -0.064115, -0.064099, 0.062113, 0.062129, 1.788830, 1.788400, 
    1.918441, 1.918905, 3.645452, 3.644571, -3.646416, -3.646708, 
    -1.919998, -1.919844, -1.789454, -1.789597, -0.062733, -0.062727, 
    0.061814, 0.061819, 1.788683, 1.788540, 1.918871, 1.919025, 
    3.645736, 3.645443, -3.646562, -3.646270, -1.919698, -1.919852, 
    -1.789413, -1.789270, -0.062544, -0.062549, 0.062876, 0.062871, 
    1.789598, 1.789741, 1.919874, 1.919720, 3.646292, 3.646584, 
    -3.645853, -3.644972, -1.918842, -1.919306, -1.789367, -1.788937, 
    -0.062651, -0.062666, 0.063143, 0.063128, 1.789415, 1.789845, 
    1.919878, 1.919413, 3.645543, 3.646424
])

_CCD_LAYOUT_Y = np.array([
    -3.727898, -2.001758, -2.004785, -3.731333, -3.729368, -2.002803, 
    -2.003812, -3.730512, -3.730976, -2.004276, -2.003269, -3.729834, 
    -3.731505, -2.004957, -2.001932, -3.728073, -1.816060, -0.089749, 
    -0.090622, -1.817335, -1.816611, -0.089881, -0.090172, -1.817035, 
    -1.817472, -0.090609, -0.090319, -1.817048, -1.817584, -0.090871, 
    -0.089998, -1.816309, 0.090679, 1.816989, 1.818266, 0.091552, 
    0.091155, 1.817884, 1.818309, 0.091446, 0.090876, 1.817739, 
    1.817315, 0.090586, 0.091290, 1.818003, 1.816728, 0.090417, 
    2.002667, 3.728808, 3.732241, 2.005694, 2.003694, 3.730258, 
    3.731401, 2.004701, 2.003834, 3.730533, 3.729391, 2.002826, 
    2.004674, 3.731221, 3.727789, 2.001648
])

# Maximum angular distance from field center to edge (degrees)
_FIELD_RADIUS = 5.66

# Cache for field data
_FIELDS_CACHE = {}


def load_ztf_fields(fields_file=None):
    """
    Load ZTF field data from file.

    Returns
    -------
    fieldno, ra, dec : arrays
        Field numbers and coordinates (RA, Dec in degrees)
    """

    fields_file = fields_file if fields_file is not None else get_fields_path()
    
    # Check cache
    cache_key = str(fields_file)
    if cache_key in _FIELDS_CACHE:
        return _FIELDS_CACHE[cache_key]
    
    # Load from file
    data = np.loadtxt(
        fields_file,
        usecols=(0, 1, 2),
        dtype=[('fieldno', int), ('ra', float), ('dec', float)]
    )
    
    result = (data['fieldno'], data['ra'], data['dec'])
    _FIELDS_CACHE[cache_key] = result
    
    return result


def get_field_ids(ra, dec, fields_file=None):
    """
    Find ZTF fields containing the given coordinates.
    
    Parameters
    ----------
    ra, dec : float
        Object coordinates in degrees
    fields_file : str or Path, optional
        Path to ZTF_Fields.txt
    
    Returns
    -------
    list of tuples
        [(field_id, ccd, quadrant), ...] for all matching fields
    """
    fieldno, ra_all, dec_all = load_ztf_fields(fields_file)
    
    deg = np.pi / 180.0
    ra_rad = ra * deg
    dec_rad = dec * deg
    ra_all_rad = ra_all * deg
    dec_all_rad = dec_all * deg
    
    results = []
    
    for i in range(len(fieldno)):
        # Quick angular distance check
        adist = ang_dist(ra_rad, dec_rad, ra_all_rad[i], dec_all_rad[i])
        if adist >= _FIELD_RADIUS:
            continue
        
        # Project onto field's tangent plane
        x, y = orthographic_projection(ra_rad, dec_rad, ra_all_rad[i], dec_all_rad[i])
        
        # Check if within any CCD
        ccd, quad = _inside_polygon(x, y, _CCD_LAYOUT_X, _CCD_LAYOUT_Y)
        if ccd is not None and quad is not None:
            results.append((fieldno[i], ccd, quad))
    
    return results

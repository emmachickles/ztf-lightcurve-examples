"""
ZTF Light Curve Extraction Utilities
------------------------------------

Lightweight wrapper for extracting ZTF light curves from matchfiles
stored on Engaging. Designed to integrate with the Periodicity Embeddings
project workflow.

Author: Emma Chickles (adapted from legacy code)
"""

import numpy as np
import h5py
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd 
import os
from ztf_lightcurves.ztf import get_field_ids, ang_dist

# Root directory of matchfiles on Engaging
DEFAULT_DATA_ROOT = Path("/orcd/data/kburdge/001/ZTF/matchfiles")


def get_h5_filename(field, ccd, quad, filt, data_root=None):
    """
    Return file path of the ZTF matchfile HDF5.

    Parameters
    ----------
    field : int
    ccd : int
    quad : int
    filt : str ('g', 'r', 'i')
    data_root : Path or None

    Returns
    -------
    Path
    """
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT

    data_root = Path(data_root)
    fname = f"data_{field:04d}_{ccd:02d}_{quad:d}_z{filt}.h5"
    fname = data_root / f"{field:04d}" / fname
    if os.path.exists(fname):
        return fname
    else:
        return None

def _iter_h5_filenames_for_target(ra, dec, filts):
    """
    Yield candidate H5 candidates for all (field,ccd,quad) that cover (ra,dec)
    and for each requested filter.
    """
    field_ids = get_field_ids(ra, dec)
    for (field, ccd, quad) in field_ids:
        for filt in filts:
            fname = get_h5_filename(field, ccd, quad, filt)
            if fname:
                yield (field, ccd, quad, filt, fname)
            

def _find_best_match(ra_deg, dec_deg, fname):
    with h5py.File(fname, "r") as h5:
        sources = h5["data"]["sources"][:]
        
    sep_deg = ang_dist(ra_deg, dec_deg, sources["ra"], sources["decl"])
    idx = np.argmin(sep_deg)
    sep_arcsec = sep_deg[idx] * 3600.0
    
    source_dict = {name: sources[name][idx].item() for name in sources.dtype.names}
    
    return idx, sep_arcsec, source_dict

def _build_lightcurve_df(fname, idx):
    
    with h5py.File(fname, "r") as h5:
        exp = h5["data"]["exposures"][:]
        
        Nexp = len(exp)
        row_idx = np.arange(Nexp, dtype=np.int64) + idx * Nexp
    
        src = h5["data"]["sourcedata"][row_idx] # (Nexp,)
        
    exp_df = pd.DataFrame.from_records(exp)
    src_df = pd.DataFrame.from_records(src)

    return pd.concat([exp_df, src_df], axis=1)


def get_lightcurve(ra, dec, tol=1.0, filt=["g", "r", "i"]):
    """
    Parameters
    ----------
    filt : str or list of str
        'g' or ['g', 'r'] etc.
    Returns
    -------
    DataFrame: concatenated per-exposure rows 
    """
    
    filts = [filt] if isinstance(filt, str) else list(filt)
    lc_parts = []
    src_parts = []
    
    for field, ccd, quad, filt, fname in _iter_h5_filenames_for_target(ra, dec, filts):

        idx, sep_arcsec, src_dict = _find_best_match(ra, dec, fname)
        
        if sep_arcsec > tol:
            continue
        
        # light curve table for this match
        lc_df = _build_lightcurve_df(fname, idx)
        lc_parts.append(lc_df)
        
        src_dict["field"] = field
        src_dict["ccd"] = ccd
        src_dict["quad"] = quad
        src_dict["filt"] = filt
        src_dict["sep_arcsec"] = sep_arcsec
        src_parts.append(src_dict)
        
    source_dict = {k: [d[k] for d in src_parts] for k in src_parts[0]}
    lc_merged = pd.concat(lc_parts, ignore_index=True)
    
    if len(set(source_dict["gaia_id"])) > 1:
        raise ValueError(f"Multiple Gaia IDs found: {set(source_dict['gaia_id'])}")
            
    return {"source": source_dict,
            "lc": lc_merged}
        
        
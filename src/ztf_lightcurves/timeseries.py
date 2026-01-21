import numpy as np

def bjd_convert(time, ra, dec, date_format='mjd', telescope='Palomar', scale='tcb'):

    from astropy.time import Time
    from astropy.coordinates import EarthLocation, SkyCoord

    # Create sky coordinate
    coord = SkyCoord(ra, dec, unit="deg")

    # Create Astropy Time object
    time_obj = Time(time, format=date_format, scale='utc')

    # Convert to desired timecale
    time_scaled = time_obj.tcb if scale == 'tcb' else time_obj.tdb

    # Get observatory location
    observatory = EarthLocation.of_site(telescope)

    # Calculate light travel time correction for barycentric motion
    correction = time_scaled.light_travel_time(coord, kind='barycentric', location=observatory)

    # Apply barycentric correction
    bjd = time_scaled + correction

    return bjd.mjd

def phase_fold(time, period, period_derivative=0, reference_epoch=None):
    """
    Phase-fold time series data accounting for period derivative.
    
    Parameters
    ----------
    time : array-like
        Time values to phase-fold
    period : float
        Orbital/rotational period at reference epoch
    period_derivative : float, optional
        Period derivative dP/dt (default: 0)
    reference_epoch : float, optional
        Reference epoch for phase folding. If None, uses minimum time (default: None)
        
    Returns
    -------
    phases : ndarray
        Phases in range [0, 1)
        
    Notes
    -----
    Accounts for period evolution with quadratic phase model:
    phi = (dt - 0.5*Pdot/P*dt^2) / P where dt = t - t0
    """
    time = np.asarray(time)
    
    if reference_epoch is None:
        reference_epoch = np.min(time)
    
    # Time relative to reference epoch
    dt = time - reference_epoch
    
    # Phase evolution with period derivative correction
    phases = ((dt - 0.5 * period_derivative / period * dt**2) % period) / period
    
    return phases

def bin_phase_folded_data(time, flux, flux_err, period, period_derivative=0, 
                          reference_epoch=None, num_bins=500, num_cycles=3, normalization=False):
    """
    Bin phase-folded light curve data with inverse-variance weighted averaging.
    
    Parameters
    ----------
    time : array-like
        Time values of observations
    flux : array-like  
        Flux/magnitude measurements
    flux_err : array-like
        Uncertainties in flux measurements
    period : float
        Orbital/rotational period for phase folding
    period_derivative : float, optional
        Period derivative dP/dt (default: 0)
    reference_epoch : float, optional
        Reference epoch for phase-folding. If None, uses min(time) (default: None)
    num_bins : int, optional
        Number of phase bins (default: 500)
    num_cycles : int, optional
        Number of cycles to replicate for plotting: 1, 2, or 3 (default: 3)
    normalization : str or False, optional
        Normalization method: 'median', 'min', 'max', 'mean', or False (default: False)
        
    Returns
    -------
    binned_lc : dict
        Dictionary with keys 'phase', 'flux', 'flux_err' containing arrays
        
    Notes
    -----
    Uses inverse-variance weighting for bin averages. Empty bins are filled with NaN.
    """
    # Phase fold the data
    phases = phase_fold(time, period, period_derivative, reference_epoch)

    # Create phase bins
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    binned_lc = []
    
    for i, phase_center in enumerate(bin_centers):
        # Find data points in this phase bin
        phase_min = bin_edges[i]
        phase_max = bin_edges[i + 1]
        
        mask = (phases >= phase_min) & (phases < phase_max)
        
        if not np.any(mask):
            # Empty bin - use NaN values
            binned_lc.append([phase_center, np.nan, np.nan])
            continue
            
        # Inverse-variance weighted average
        bin_flux = flux[mask]
        bin_flux_err = flux_err[mask]
        
        # Filter out invalid errors (zero, negative, or NaN)
        valid = (bin_flux_err > 0) & np.isfinite(bin_flux_err) & np.isfinite(bin_flux)
        
        if not np.any(valid):
            # No valid data in this bin
            binned_lc.append([phase_center, np.nan, np.nan])
            continue
        
        bin_flux = bin_flux[valid]
        bin_flux_err = bin_flux_err[valid]
        
        weights = 1.0 / (bin_flux_err**2)
        
        weighted_mean = np.sum(bin_flux * weights) / np.sum(weights)
        weighted_error = np.sqrt(1.0 / np.sum(weights))
        
        binned_lc.append([phase_center, weighted_mean, weighted_error])
    
    binned_lc = np.array(binned_lc)

    # Apply normalization if requested
    if normalization:
        valid_flux = binned_lc[:, 1][np.isfinite(binned_lc[:, 1])]
        if len(valid_flux) == 0:
            raise ValueError("No valid flux values for normalization")
            
        norm_methods = {
            'median': np.median,
            'min': np.min,
            'max': np.max,
            'mean': np.mean
        }
        
        norm_method = normalization.lower()
        if norm_method not in norm_methods:
            raise ValueError(f"Unknown normalization method: {normalization}. "
                           f"Choose from {list(norm_methods.keys())}")
            
        norm_factor = norm_methods[norm_method](valid_flux)
        binned_lc[:, 1] /= norm_factor
        binned_lc[:, 2] /= norm_factor
        
    # Replicate cycles for display
    if num_cycles == 1:
        pass  # Keep as is
    elif num_cycles == 2:
        cycle_prev = binned_lc.copy()
        cycle_prev[:, 0] -= 1.0
        binned_lc = np.vstack([cycle_prev, binned_lc])
    elif num_cycles == 3:
        cycle_prev = binned_lc.copy()
        cycle_prev[:, 0] -= 1.0
        cycle_next = binned_lc.copy()
        cycle_next[:, 0] += 1.0
        binned_lc = np.vstack([cycle_prev, binned_lc, cycle_next])
    else:
        raise ValueError(f"num_cycles must be 1, 2, or 3, got {num_cycles}")
    
    # Return as dictionary
    return {
        'phase': binned_lc[:, 0],
        'flux': binned_lc[:, 1],
        'flux_err': binned_lc[:, 2]
    }

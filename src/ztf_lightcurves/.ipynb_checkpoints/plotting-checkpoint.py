import numpy as np
import matplotlib.pyplot as plt

from .timeseries import bin_phase_folded_data  # adjust import if your package layout differs


_FILTERID_STYLE = {
    1: ("g", "tab:green"),
    2: ("r", "tab:red"),
    3: ("i", "tab:purple"),
}

def plot_raw_lightcurve(qry, time_col="bjd", flux_col="flux", err_col="flux_err",
                        filter_col="filterid", flag_col="flag",
                        drop_bad_flags=True, ax=None):
    """
    Plot raw light curve, coloring points by ZTF filterid:
      1 -> g, 2 -> r, 3 -> i
    """
    df = qry["lc"].copy()

    # optional basic flag cleaning (adjust to your flag semantics if needed)
    if drop_bad_flags and flag_col in df.columns:
        df = df[df[flag_col] == 0]

    if ax is None:
        _, ax = plt.subplots()

    # plot each filter separately so legend is clean
    for fid, (lab, col) in _FILTERID_STYLE.items():
        m = (df[filter_col] == fid) & np.isfinite(df[time_col]) & np.isfinite(df[flux_col])
        if err_col in df.columns:
            m &= np.isfinite(df[err_col])

        if not np.any(m):
            continue

        ax.errorbar(df.loc[m, time_col], df.loc[m, flux_col],
                    yerr=df.loc[m, err_col] if err_col in df.columns else None,
                    fmt=".", ms=3, lw=0.8, capsize=0,
                    color=col, alpha=0.9)

    ax.set_xlabel(time_col)
    ax.set_ylabel(flux_col)
    return ax


def plot_phase_folded_binned_lightcurve(
    qry,
    period,
    period_derivative=0.0,
    reference_epoch=None,
    num_bins=200,
    num_cycles=3,
    normalization=False,
    time_col="bjd",
    flux_col="flux",
    err_col="flux_err",
    filter_col="filterid",
    flag_col="flag",
    drop_bad_flags=True,
    ax=None,
):
    """
    Plot phase-folded, binned light curve. Bins each filter separately, then overlays.

    Parameters
    ----------
    qry : dict
        Output of get_lightcurve, with keys 'source' (dict) and 'lc' (DataFrame)
    period : float
        Period in same time units as time_col (e.g., days if bjd is MJD)
    period_derivative : float
        dP/dt in same units (e.g., days/day)
    reference_epoch : float or None
        Reference epoch t0 for phase folding (same units as time)
    num_bins : int
        Number of phase bins per cycle (before replication)
    num_cycles : int
        1, 2, or 3 (for display -1..0..+1)
    normalization : str or False
        'median','mean','min','max', or False
    """

    df = qry["lc"].copy()

    if drop_bad_flags and flag_col in df.columns:
        df = df[df[flag_col] == 0]

    if ax is None:
        _, ax = plt.subplots()

    # Plot per filter (binned separately), then overlay
    any_plotted = False
    for fid, (lab, col) in _FILTERID_STYLE.items():
        m = (df[filter_col] == fid)
        if not np.any(m):
            continue

        t = np.asarray(df.loc[m, time_col], dtype=float)
        y = np.asarray(df.loc[m, flux_col], dtype=float)
        yerr = np.asarray(df.loc[m, err_col], dtype=float) if err_col in df.columns else None

        if yerr is None:
            raise ValueError(f"'{err_col}' column is required for inverse-variance binning")

        # finite mask
        ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
        if not np.any(ok):
            continue

        b = bin_phase_folded_data(
            time=t[ok],
            flux=y[ok],
            flux_err=yerr[ok],
            period=period,
            period_derivative=period_derivative,
            reference_epoch=reference_epoch,
            num_bins=num_bins,
            num_cycles=num_cycles,
            normalization=normalization,
        )

        ax.errorbar(
            b["phase"],
            b["flux"],
            yerr=b["flux_err"],
            fmt=".",
            ms=3,
            lw=0.8,
            capsize=0,
            color=col,
            alpha=0.9,
        )
        any_plotted = True

    if not any_plotted:
        raise ValueError("No valid data to plot after filtering (flags/NaNs/errors).")

    ax.set_xlabel("Phase")
    ax.set_ylabel(flux_col)
    return ax
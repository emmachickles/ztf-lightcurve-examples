# Load ZTF light curve of ZTF J153932.16+502738.8

ra  = 234.8839713056
dec = 50.4607560776

from ztf_lightcurves.ztf_lc import get_lightcurve
qry = get_lightcurve(ra, dec)

## Inspect structure of output

print(qry.keys())
print(type(qry["source"]), list(qry["source"].keys()))
print(type(qry["lc"]), qry["lc"].shape,qry["lc"].columns.tolist())
qry["lc"].head()

## Inspect lightcurve

from ztf_lightcurves.plotting import plot_raw_lightcurve, plot_phase_folded_binned_lightcurve
from ztf_lightcurves.config import out_path
import matplotlib.pyplot as plt

plot_raw_lightcurve(qry)
fname = out_path("ZTFJ1539+5027_raw.png")
plt.savefig(fname)

period = 414.7915404 / 86400
ax = plot_phase_folded_binned_lightcurve(qry, period=period, num_bin s=100)
fname =out_path("ZTFJ1539+5027_folded.png")
plt.savefig(fname)
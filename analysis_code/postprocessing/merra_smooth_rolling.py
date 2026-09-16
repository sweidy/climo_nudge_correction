import os
import re
import numpy as np
import xarray as xr
from pathlib import Path

# ========= user settings =========
#out_dir = "/n/home04/sweidman/holylfs06/MERRA2_OG/MERRA2_f19/monave30/"
#in_dir = Path(out_dir)
#pattern = re.compile(r"^MERRA2_avg_\d{4}_\d{5}\.nc$")
#out_dir="/n/home04/sweidman/holylfs06/ERA5/ERA5_f19/monave/"
#in_dir=Path(out_dir)
#pattern=re.compile(r"^ERA5_avg_\d{4}_\d{5}\.nc$")
# out_dir = "/n/home04/sweidman/holylfs06/CESM2_corrector_out/Run/archive/spcam_control_6hr/rolling"
# in_dir = Path("/n/home04/sweidman/holylfs06/CESM2_corrector_out/Run/archive/spcam_control_6hr/rolling")
# pattern = re.compile(r"^spcam_control_6hr.cam.h2.avg-\d{2}-\d{2}-\d{5}\.nc$")
out_dir = "/n/home04/sweidman/holylfs06/IC_CESM2"
in_dir = Path("/n/home04/sweidman/holylfs06/IC_CESM2")
pattern = re.compile(r"^nudge30_win_1_i5.\d{2}-\d{2}-\d{5}\.nc$")

def out_name(doy_1based: int, hbin: int) -> str:
    month, day = doy_to_month_day(doy_1based)
    sec = hbin * secs_per_bin
    #return f"MERRA2_rolled_{month:02}{day:02}_{sec:05}.nc"
    #return f"ERA5_rolled_{month:02}{day:02}_{sec:05}.nc"
    #return f"spcam_control_6hr_rolled.{month:02}-{day:02}-{sec:05}.nc"
    return f"nudge30_win_1_i5_rolled.{month:02}-{day:02}-{sec:05}.nc"

#variables = ['U','V','T','Q']
variables = ['UDIFF','VDIFF','SDIFF','QDIFF']
nhour=4
ndoy=365
hour = np.tile(np.arange(nhour), ndoy) 
dayofyear = np.repeat(np.arange(1,ndoy+1),nhour)
window=31
half = window//2

files = [
    str(p) for p in in_dir.iterdir()
    if pattern.match(p.name)
]
files.sort()
print(files[0])

my_time = np.arange(0,ndoy,1/nhour) 
assert len(files) == len(my_time), (len(files), len(my_time))

# time bookkeeping for naming outputs
daylist = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=int)
cum = np.cumsum(daylist)
secs_per_bin = int(24 / nhour * 3600)  # 10800 for nhour=8

def doy_to_month_day(doy_1based: int):
    m = int(np.searchsorted(cum, doy_1based, side="left")) + 1
    day_in_month = int(doy_1based - (cum[m-2] if m > 1 else 0))
    return m, day_in_month

# ========= I/O helpers =========
def open_vars_one(fp: str) -> xr.Dataset:
    """
    Open one file, take time=0, keep only requested vars.
    """
    ds = xr.open_dataset(fp, decode_times=False, engine="netcdf4")
    if "time" in ds.dims:
        ds = ds.isel(time=0, drop=True)

    ds = ds.drop_vars("time", errors="ignore")
    ds = ds[variables]

    return ds
    # ds = ds.isel(time=0, drop=True)#.expand_dims(time=[0]).assign_coords(time=("time", [0])) 
    # ds = ds[variables]
    # #ds = ds.reset_coords(drop=True)
    # return ds

def window_indices(doy_1based: int, hbin: int) -> np.ndarray:
    doy0 = doy_1based - 1
    doys0 = (np.arange(doy0 - half, doy0 + half + 1) % ndoy).astype(int)  # length=window

    # map each doy to file index at the same hour
    # t = doy*nhour + hbin  (with doy 0-based)
    return doys0 * nhour + hbin

for doy in range(1, ndoy + 1):
    for hbin in range(0,nhour):
        idx = window_indices(doy, hbin)

        dsets = [open_vars_one(files[j]) for j in idx]

        stack = xr.concat(dsets, dim="win")
        rolled = stack.mean(dim="win", keep_attrs=True)#.expand_dims(time=[0])
            
        rolled = rolled.drop_vars("time", errors="ignore")

        # close file handles ASAP
        for dsj in dsets:
            dsj.close()
        stack.close()

        out_path = os.path.join(out_dir, out_name(doy, hbin))
        rolled.to_netcdf(out_path)

        rolled.close()
        del dsets, stack, rolled

    print(f"wrote {out_path}")

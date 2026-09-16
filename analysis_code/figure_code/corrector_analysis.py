import os
import glob
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import random 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter

# Cartopy components for geographic projections
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import cartopy.mpl.ticker as cticker

# External climate computation library
import geocat.comp as gcomp

# =====================================================================
# PART 1: USER-DEFINED HELPER FUNCTIONS
# =====================================================================

def open_monthly_model(path_str, years=np.arange(1980,2000), mons=np.arange(1,13), var_list=['U','V','T','Q','PRECT','PS'], regrid=False, extrap=True):
    # Read in monthly model data and return as single dataset

    run_list_mon = []
    print(path_str)
    for m in mons:
        
        run_list = []
        for y in years:
            path2 = f'.cam.h0.{y:04}-{m:02}.nc'
            ds = xr.open_dataset(path_str + path2)

            if 'PRECT' in var_list:
                try:
                    run_list.append(ds[var_list])
                except KeyError:
                    fallback_var_list = [v for v in var_list if v != 'PRECT'] + ['PRECC', 'PRECL']
                    run_list.append(ds[fallback_var_list])
            else:
                run_list.append(ds[var_list])    
            
        runs = xr.concat(run_list, dim='time')
        run_list_mon.append(runs.mean(dim='time'))

    run_mon_ds = xr.concat(run_list_mon, pd.Index(mons, name='month'))

    if 'PRECC' in run_mon_ds.data_vars:
        run_mon_ds['PRECT'] = run_mon_ds['PRECC'] + run_mon_ds['PRECL'] 
        
    if regrid:
        print('regrid model to plev')
        return regrid_plev(run_mon_ds, extrap=extrap)
    else:
        return run_mon_ds

def open_monthly_merra(path_str, mons, climo_set="monave",regrid=False, extrap=True):
    # read in monthly (climatology) MERRA data and return single dataset
    
    run_list_mon = []
    
    for m in mons:
        path2 = f'{climo_set}/MERRA2_avg2_{m:02}.nc'
        #path2 = f'monave/ERA5_avg2_{m:02}.nc'
        
        run_list_mon.append(xr.open_dataset(path_str + path2, decode_times=False)[['U','V','T','Q','PS']])
    
    #return xr.concat(run_list_mon, pd.Index(mons, name='month'))
    mon_concat = xr.concat(run_list_mon, dim='time')
    mon_concat['time'] = pd.Index(mons)

    mon_concat_ds = mon_concat.rename({'time': 'month'})
    
    if regrid:
        print('regrid MERRA to plev')
        return regrid_plev(mon_concat_ds, extrap=extrap)
    else: 
        return mon_concat_ds

def open_monthly_merra_sfc(path_str, mons, climo_set="monave"):
    # read in monthly (climatology) MERRA data and return single dataset
    
    ts_list_mon = []
    flux_list_mon = []
    omega_list_mon = []

    if climo_set == "monave":
        sfc_set = "sfc_monthly"
    elif climo_set == "monave30":
        sfc_set = "sfc_monthly30"
    
    for m in mons:
        path_ts = f'{sfc_set}/M2IUNXASM.5.12.4:MERRA2_{m:02}.nc'
        path_flx = f'{sfc_set}/M2TUNXFLX.5.12.4:MERRA2_{m:02}.nc'
        path_omega = f'{sfc_set}/M2TMNXSLV.5.12.4:MERRA2_{m:02}.nc' 
        
        ts_list_mon.append(xr.open_dataset(path_str + path_ts))
        flux_list_mon.append(xr.open_dataset(path_str + path_flx))
        omega_list_mon.append(xr.open_dataset(path_str + path_omega))
    
    return xr.concat(ts_list_mon, pd.Index(mons, name='month')), xr.concat(flux_list_mon, pd.Index(mons, name='month')), xr.concat(omega_list_mon, pd.Index(mons, name='month'))

def open_monthly_ceres_toa(mons):
    # read in monthly (climatology) CERES data and return single dataset
    
    ceres = xr.open_dataset('/n/home04/sweidman/holylfs06/CERES_EBAF-TOA_Ed4.2.1_Subset_CLIM01-CLIM12.nc')

    ceres['SWCF'] = ceres['toa_sw_clr_c_clim'] - ceres['toa_sw_all_clim']
    ceres['LWCF'] = ceres['toa_lw_clr_c_clim'] - ceres['toa_lw_all_clim']
    ceres = ceres.rename({'cldarea_total_daynight_clim':'CLDTOT','ctime':'month'})
    ceres['CLDTOT'] = ceres['CLDTOT']/100

    ceres = ceres[['SWCF','LWCF','CLDTOT']]
    
    return ceres.sel(month=mons) 

def open_monthly_tendencies(path_str, mons, regrid=False, extrap=True):
    # Read in monthly mean tendencies and return single dataset
    
    run_list_mon = []
    
    for m in mons:
        path2 = f'.{m:02}.nc'
        
        try:
            run_list_mon.append(xr.open_dataset(path_str + path2)[['UDIFF','VDIFF','SDIFF','QDIFF']])
        except KeyError:
            run_list_mon.append(xr.open_dataset(path_str + path2)[['Running_nudge_U','Running_nudge_V','Running_nudge_T','Running_nudge_Q']])
    
    run_ds = xr.concat(run_list_mon, pd.Index(mons, name='month'))

    if 'Running_nudge_U' in run_ds.data_vars:
        run_ds = xr.Dataset({
            "UDIFF": run_ds["Running_nudge_U"]*21600,
            "VDIFF": run_ds["Running_nudge_V"]*21600,
            "SDIFF": run_ds["Running_nudge_T"]*21600*1004.9,
            "QDIFF": run_ds["Running_nudge_Q"]*21600,
            })

    if regrid:
        print('regrid tendencies to plev')
        return regrid_plev(run_ds, extrap=extrap)
    else: 
        return run_ds

def weighted_monthly_mean(ds, time_dim="month"):
    # take monthly mean of variable, weighted by days in month
    
    days_per_mon = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # Get month numbers (1..12) from the coordinate
    coord = ds[time_dim]
    if hasattr(coord, "dt"):
        months = coord.dt.month
    else:
        months = xr.DataArray(coord.astype(int), dims=coord.dims, coords=coord.coords, name="month")

    # Build weights by mapping month -> days
    w_vals = np.array([days_per_mon[m - 1] for m in months.values])
    weights = xr.DataArray(w_vals, dims=coord.dims, coords=coord.coords, name="month_days")

    # xarray handles normalization internally; NaNs are handled per-variable
    return ds.weighted(weights).mean(dim=time_dim)

def regrid_plev(ds, target_plev=np.array([1, 10, 50] + list(range(100, 901, 50)) + list(range(925, 1001, 25)), dtype=float)*100,
                extrap=True):
    # regrid model level data to pressure levels using standard CAM grid information. Specifiy target
    # pressure levels as needed

    var_list = ds.data_vars
    
    temp_grid_file = xr.open_dataset('/n/home04/sweidman/holylfs06/MERRA2_OG/MERRA2_f19/monave/MERRA2_avg2_01.nc',decode_times=False)
    temp_grid_file = temp_grid_file.squeeze('scalar')

    cam_phis_file = xr.open_dataset('/n/holystore01/INTERNAL_REPOS/CLIMATE_MODELS/cesm_2_1/inputdata/atm/cam/topo/USGS-gtopo30_1.9x2.5_remap_c050602.nc')
    phi_sfc = cam_phis_file['PHIS'].assign_coords({'lat': ds['lat'], 'lon': ds['lon']})

    regridded_dict = {}
    
    for var in var_list:

        if 'lev' in ds[var].dims:
            # Run GeoCAT hybrid-to-pressure conversion
            if var == 'T':
                extrap_type = "temperature"
            elif var == 'Z3':
                extrap_type = "geopotential"
            else:
                extrap_type = "other"

            if extrap_type == 'other':
                plev_data = gcomp.interp_hybrid_to_pressure(
                    data=ds[var], 
                    ps=ds['PS'],
                    p0=temp_grid_file['P0'],
                    hyam=temp_grid_file['hyam'],
                    hybm=temp_grid_file['hybm'],
                    new_levels=target_plev,
                    extrapolate=extrap,
                    variable=extrap_type
                )
            else:
                t_bot = ds['T'].isel(lev=-1)
                plev_data = gcomp.interp_hybrid_to_pressure(
                    data=ds[var], 
                    ps=ds['PS'],
                    p0=temp_grid_file['P0'],
                    hyam=temp_grid_file['hyam'],
                    hybm=temp_grid_file['hybm'],
                    new_levels=target_plev,
                    extrapolate=extrap,
                    variable=extrap_type,
                    t_bot=t_bot,
                    phi_sfc=phi_sfc
                )
            
            plev_data.attrs = ds[var].attrs
            
            regridded_dict[var] = plev_data
        else:
            regridded_dict[var] = ds[var]
    
    ds_pressure = xr.Dataset(regridded_dict)
    
    # Update pressure coordinate attributes for clean documentation
    if 'plev' in ds_pressure.dims:
        ds_pressure['plev'] = ds_pressure['plev'] / 100.0
        ds_pressure['plev'].attrs = {
            'long_name': 'pressure',
            'units': 'hPa',
            'positive': 'down'
        }
    return ds_pressure.compute()

def open_daily_regex(filedir, case_type, years, months, varsel=None, regrid=False, extrap=True):
    # open daily climatology with regex and return mfdataset

    year_start = years[0]
    year_end = years[-1]

    if case_type == 'model':
        #check for files
        files = glob.glob(filedir+'*.cam.h1.*')
        files = np.array(sorted(files))
    
        date_range_full = xr.date_range(start='1980', periods=len(files), freq='D', calendar='noleap')
        files_range = files[(date_range_full.year >= year_start) & (date_range_full.year <= year_end)]
        date_range = date_range_full[(date_range_full.year >= year_start) & (date_range_full.year <= year_end)]

        print(files_range[0], files_range[-1])
        
        ds = xr.open_mfdataset(files_range, chunks = {})
        ds = ds.assign_coords(time=(date_range))
        
    elif case_type == 'MERRA':
        #check for files
        files = glob.glob(filedir+'MERRA2_*_0.nc')
        #files = glob.glob(filedir+'ERA5_*_0.nc')
        files = np.array(sorted(files))

        date_range_full = xr.date_range(start='1980', periods=len(files), freq='D', calendar='noleap')
        files_range = files[(date_range_full.year >= year_start) & (date_range_full.year <= year_end)]
        date_range = date_range_full[(date_range_full.year >= year_start) & (date_range_full.year <= year_end)]

        print(files_range[0], files_range[-1])

        ds = xr.open_mfdataset(sorted(files_range), concat_dim='time', combine='nested', 
                               coords='minimal', decode_times=False, chunks = {})[['U','V','T','Q','PS','PHIS']]
        ds = ds.assign_coords(time=(date_range))

    ds=ds.sel(time=ds.time.dt.month.isin(months))

    if varsel is not None:
        ds = ds[varsel]
        
    if regrid:
        print('regrid daily to plev')
        return regrid_plev(ds, extrap=extrap)
    else:
        return ds

def calculate_transients(ds):

    len_lat = len(ds.lat)
    len_lon = len(ds.lon)
    
    ds_chunked = ds.chunk({'time':90, 'lat':len_lat, 'lon':len_lon})
    monthly = ds_chunked.groupby('time.month').mean(dim='time').persist()

    ds_anom = ds_chunked.groupby('time.month') - monthly

    transients = xr.Dataset(coords=ds_anom.coords)

    print('...doing calcs...')
    transients['UpUp'] = ds_anom['U']*ds_anom['U']
    transients['UpVp'] = ds_anom['U']*ds_anom['V']
    transients['VpVp'] = ds_anom['V']*ds_anom['V']
    transients['VpQp'] = ds_anom['V']*ds_anom['Q']
    transients['VpTp'] = ds_anom['V']*ds_anom['T']
    transients['EKE'] = 0.5*(ds_anom['U']**2 + ds_anom['V']**2)


    # Monthly climatological transient eddy quantities
    transients = transients.groupby("time.month").mean(dim="time").persist()

    # Annual mean from monthly means, weighted by days per month
    # month_length = ds_anom.time.dt.days_in_month
    # weights = (month_length.groupby("time.season") / month_length.groupby("time.season").sum())
    # ds_anom = (ds_anom * weights).groupby("time.season").sum(dim="time").persist()

    return transients

def apply_regional_mask(da, region_name, lat_coord, landfrac_da, lf_scale):
    """
    Applies latitude slices, tropical/extratropical zones, and land/ocean fractional 
    masks based on climate diagnostics criteria.
    """
    # Standard baseline: truncate grid point edges to avoid polar singularities
    masked_da = da.sel(lat=slice(-87, 87))
    lats = lat_coord.sel(lat=slice(-87, 87))

    # Extract specific geographic domain windows
    if region_name == 'global':
        pass  # Already bound between -87 and 87
    elif region_name == 'tropics':
        masked_da = masked_da.where((lats < 25) & (lats > -25))
    elif region_name == 'extratropics':
        masked_da = masked_da.where((lats > 25) | (lats < -25))
    elif region_name == 'land':
        masked_da = masked_da.where(landfrac_da >= lf_scale)
    elif region_name == 'ocean':
        masked_da = masked_da.where(landfrac_da <= lf_scale)
    else:
        raise ValueError(f"Unknown target evaluation region: {region_name}")
        
    return masked_da.squeeze()

def compute_weighted_rmse(obs, model):
    """
    Computes horizontal area-weighted Root Mean Square Error (RMSE).
    """

    obs_avg = weighted_monthly_mean(obs)
    model_avg = weighted_monthly_mean(model)

    lat = obs.lat
    cos_lat_weights = np.cos(lat*np.pi/180)
    squared_error = (model_avg - obs_avg) ** 2
    
    mean_error = squared_error.weighted(cos_lat_weights).mean(dim=['lat', 'lon'], skipna=True)
    
    return np.sqrt(mean_error)

def compute_weighted_rmse_season(obs, model):

    lat_weights = np.cos(np.deg2rad(obs.lat))

    if obs.sizes['month'] == 12:
        seasons = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
        month_days = xr.DataArray([31,28,31,30,31,30,31,31,30,31,30,31],
                                  coords={'month': np.arange(1,13)}, dims='month')

        seasonal_errors, season_days = [], []
        for season, smonths in seasons.items():
            days = month_days.sel(month=smonths)
            obs_season = weighted_monthly_mean(obs.sel(month=smonths))
            model_season = weighted_monthly_mean(model.sel(month=smonths))

            seasonal_errors.append(((model_season - obs_season)**2).weighted(lat_weights).mean(['lat','lon'], skipna=True))
            season_days.append(float(days.sum()))

        seasonal_errors = xr.concat(seasonal_errors, dim=xr.DataArray(list(seasons), dims='season', name='season'))
        season_weights = xr.DataArray(season_days, coords={'season': list(seasons)}, dims='season')
        mean_error = seasonal_errors.weighted(season_weights).mean('season', skipna=True)

    else:
        obs_avg = weighted_monthly_mean(obs)
        model_avg = weighted_monthly_mean(model)
        mean_error = ((model_avg - obs_avg)**2).weighted(lat_weights).mean(['lat','lon'], skipna=True)

    return np.sqrt(mean_error)

def compute_pattern_correlation(obs, model, varlist):

    levs = obs.lev
    r_arr = np.empty([len(levs), len(varlist)])

    for idx,l in enumerate(levs):
        for jdx,v in enumerate(varlist):

            a_sel = weighted_monthly_mean(obs[v].sel(lev=l, method='nearest'))
            b_sel = weighted_monthly_mean(model[v].sel(lev=l, method='nearest'))
            
            WGT = np.cos(a_sel.lat*np.pi/180)
            xyCov    = np.sum(WGT*a_sel*b_sel)
            xAnom2   = np.sum(WGT*a_sel**2)
            yAnom2   = np.sum(WGT*b_sel**2)
        
            r_arr[idx,jdx]   = xyCov/(np.sqrt(xAnom2)*np.sqrt(yAnom2))

    return xr.Dataset(data_vars={v: ("lev", r_arr[:, jdx]) for jdx, v in enumerate(varlist)},
                      coords={"lev": levs})

def convert_kgm2s_to_mmday(data):
    # convert precipitation rate data in kg/m^2/s to mm/day

    return data*86400

def convert_m_to_gz(data):
    # convert raw height to geopotential height

    return data*9.81

def calculate_bias_ds(case_str, merra_str, years, months, climo_set="monave",
                      regrid=False, sfc_var=False, toa_var=False, extrap=True):

    global DO_SURFACE
    global DO_TOA

    model_season = open_monthly_model(case_str, years = years, mons=months, regrid=regrid, extrap=extrap)
    merra_season = open_monthly_merra(merra_str, mons=months, climo_set=climo_set, regrid=regrid, extrap=extrap)

    model_diff = model_season - merra_season
    
    # precip dataset from ncar-ncep
    ncep_pr = xr.open_dataset('~/holylfs06/ncep-ncar/prate/prate.day.mean.nc')['prate']
    ncep_pr_season = ncep_pr.groupby('time.month').mean(dim='time').sel(month=months)
    ncep_pr_season = ncep_pr_season * 86400.0 # kg/m2/s to mm/day

    # precip bias
    model_interp = model_season[['PRECT']].interp(lat=ncep_pr.lat, lon=ncep_pr.lon, method='nearest')
    model_interp['PRECT'] = model_interp['PRECT'] * 86400.0 * 1000.0 # m/s to mm/day
    pr_diff = model_interp.PRECT - ncep_pr_season

    print('Mean precip bias', weighted_monthly_mean(pr_diff).mean().values)
    print('% diff', weighted_monthly_mean(pr_diff).mean().values/weighted_monthly_mean(ncep_pr_season).mean().values)

    model_merge = [model_season]
    diff_merge = [model_diff]
    obs_merge = [merra_season]

    if sfc_var:
        try:
            try: 
                model_sfc = open_monthly_model(case_str, years = years, mons=months, var_list=['TS','TMQ','TREFHT','PRECT','SHFLX','LHFLX','Z500'],regrid=False, extrap=extrap)
                model_omega = open_monthly_model(case_str, years = years, mons=months, var_list=['OMEGA','PS'],regrid=True, extrap=extrap) 
                model_sfc['OMEGA500'] = model_omega['OMEGA'].sel(plev=500) 
            except KeyError:
                model_sfc = open_monthly_model(case_str, years = years, mons=months, var_list=['TS','TMQ','TREFHT','PRECT','SHFLX','LHFLX'],regrid=False, extrap=extrap)
                model_omega = open_monthly_model(case_str, years = years, mons=months, var_list=['OMEGA','Z3','T','PS'],regrid=True, extrap=extrap) 
                model_sfc['OMEGA500'] = model_omega['OMEGA'].sel(plev=500,drop=True)
                model_sfc['Z500'] = model_omega['Z3'].sel(plev=500,drop=True)  
        except KeyError:
            print('missing surface var')
            DO_SURFACE = False
        else:
            merra_ts, merra_flx, merra_omega = open_monthly_merra_sfc(merra_str, mons=months,climo_set=climo_set)

            merra_ts_rolled = (merra_ts.assign_coords(lon=(merra_ts.lon % 360)).sortby("lon"))
            merra_flx_rolled = (merra_flx.assign_coords(lon=(merra_flx.lon % 360)).sortby("lon"))
            merra_omega_rolled = (merra_omega.assign_coords(lon=(merra_omega.lon % 360)).sortby("lon"))
            merra_ts_interp = merra_ts_rolled.interp(lat=model_sfc.lat, lon=model_sfc.lon, method='nearest') # Could change to linear
            merra_flx_interp = merra_flx_rolled.interp(lat=model_sfc.lat, lon=model_sfc.lon, method='nearest')
            merra_omega_interp = merra_omega_rolled.interp(lat=model_sfc.lat, lon=model_sfc.lon, method='nearest')

            merra_sfc = xr.merge([merra_ts_interp, merra_flx_interp, merra_omega_interp], compat='minimal')

            merra_sfc['TMQ'] = merra_sfc['TQI']+merra_sfc['TQV']+merra_sfc['TQL']
            merra_sfc = merra_sfc.rename({'T2M':'TREFHT','HFLUX':'SHFLX', 'EFLUX':'LHFLX','H500':'Z500'})
            merra_sfc['PRECT_MERRA'] = merra_sfc['PRECTOTCORR']*86400.0
            model_sfc['PRECT_MERRA'] = model_sfc['PRECT'] * 86400.0 * 1000.0

            sfc_diff = model_sfc - merra_sfc

            print('Mean MERRA precip bias', weighted_monthly_mean(sfc_diff['PRECT_MERRA']).mean().values)

            model_merge.append(model_sfc)
            diff_merge.append(sfc_diff)
            obs_merge.append(merra_sfc)

    if toa_var:
        try: 
            model_toa = open_monthly_model(case_str, years = years, mons=months, var_list=['CLDTOT','SWCF','LWCF'],regrid=False)
        except KeyError:
            DO_TOA = False
        else:
            ceres_toa = open_monthly_ceres_toa(months) 

            ceres_interp = ceres_toa.interp(lat=model_toa.lat, lon=model_toa.lon, method='nearest')

            toa_diff = model_toa - ceres_interp

            model_merge.append(model_toa)
            diff_merge.append(toa_diff)
            obs_merge.append(ceres_interp)

    model_season_full = xr.merge(model_merge, compat='minimal')
    model_diff_full = xr.merge(diff_merge, compat='minimal')
    merra_season_full = xr.merge(obs_merge, compat='minimal')

    return model_season_full, model_diff_full, model_interp, merra_season_full, pr_diff

def take_spatial_average_xr(data: xr.DataArray, orientation: str) -> xr.DataArray:
    """
    Takes average of data by latitude or longitude, specified by orientation, for each time step.
    Restrict spatial boundaries before passing into this function. 
     
    :param data: xr DataArray with dimensions [time, lat, lon]
    :param orientation: either 'lat' or 'lon' - tells function along which axis to take the spatial average. 
    'lat' will average over latitudes for creation of a lon x time plot, and vice versa
     
    :returns: xr DataArray with dimensions [time, lat] or [time, lon] if orientation is 'lon' or 'lat', respectively 
    """
     
    if orientation == 'lon':
        data_aved = data.mean(dim=orientation, skipna=True)                                                                                
    elif orientation == 'lat':
        # weight by latitude (more important for extratropics)
        weights = np.cos(np.deg2rad(data.lat))
        data_aved = data.weighted(weights).mean(dim='lat', skipna=True)
    else:
        raise ValueError("Orientation must be either 'lat' or 'lon'")
     
    return data_aved 

# =====================================================================
# PART 2: STANDARDIZED PLOTTING MODULES
# =====================================================================

def plot_lat_vs_level_bias(bias_var, merra_var, case_name, output_path):
    """Generates a 2x2 latitude-vertical cross-section of state variable biases."""
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    
    clevu = np.arange(-2, 2.1, .2)
    clevv = np.arange(-1, 1.1, .1)
    clevt = np.arange(-4, 4.5, .5)
    clevq = np.arange(-0.5, 0.55, .05) / 1000

    bias_var_latave = take_spatial_average_xr(bias_var, orientation='lon')
    merra_latave = take_spatial_average_xr(merra_var, orientation='lon')

    has_lev = 'lev' in bias_var_latave.dims
    vert_coord = bias_var_latave.lev if has_lev else bias_var_latave.plev

    # U Bias
    e = ax[0,0].contourf(bias_var_latave.lat, vert_coord, weighted_monthly_mean(bias_var_latave.U),
                         levels=clevu, cmap=plt.cm.bwr, extend='both', norm=mcolors.TwoSlopeNorm(0))
    ax[0,0].contour(merra_latave.lat, vert_coord, weighted_monthly_mean(merra_latave.U), colors='grey')
    fig.colorbar(e, ax=ax[0,0], label='m/s')
    ax[0,0].set_title('U Bias')

    # V Bias
    h = ax[0,1].contourf(bias_var_latave.lat, vert_coord, weighted_monthly_mean(bias_var_latave.V),
                         levels=clevv, cmap=plt.cm.bwr, extend='both', norm=mcolors.TwoSlopeNorm(0))
    ax[0,1].contour(merra_latave.lat, vert_coord, weighted_monthly_mean(merra_latave.V), colors='grey')
    fig.colorbar(h, ax=ax[0,1], label='m/s')
    ax[0,1].set_title('V Bias')

    # T Bias
    f = ax[1,0].contourf(bias_var_latave.lat, vert_coord, weighted_monthly_mean(bias_var_latave.T),
                         levels=clevt, cmap=plt.cm.bwr, extend='both', norm=mcolors.TwoSlopeNorm(0))
    ax[1,0].contour(merra_latave.lat, vert_coord, weighted_monthly_mean(merra_latave.T), colors='grey')
    fig.colorbar(f, ax=ax[1,0], label='K')
    ax[1,0].set_title('T Bias')

    # Q Bias
    g = ax[1,1].contourf(bias_var_latave.lat, vert_coord, weighted_monthly_mean(bias_var_latave.Q),
                         levels=clevq, cmap=plt.cm.bwr, extend='both', norm=mcolors.TwoSlopeNorm(0))
    ax[1,1].contour(merra_latave.lat, vert_coord, weighted_monthly_mean(merra_latave.Q), colors='grey')
    fig.colorbar(g, ax=ax[1,1], label='kg/kg')
    ax[1,1].set_title('Q Bias')

    ax[0,0].invert_yaxis()
    ax[0,0].set_ylabel('hPa')
    ax[1,0].set_ylabel('hPa')

    for axs in ax.flatten():
        axs.set_xlabel('Latitude')
        
    fig.suptitle(f'Bias: {case_name}', y=.95)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_lat_vs_lon_bias(bias_var, merra_var, case_name, 
                         var, levsel, clev, clab, 
                         output_path):
    """Maps horizontal biases at specific pressure surfaces for a single case."""

    has_lev = 'lev' in bias_var[var].dims
    has_plev = 'plev' in bias_var[var].dims
    if levsel is None:
        sel_dict = {}
    elif has_lev:
        sel_dict = {'lev': levsel}
    elif has_plev:
        sel_dict = {'plev': levsel}

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

    plotting_map_c4, plot_lons = add_cyclic_point(
        weighted_monthly_mean(bias_var[var]).sel(method='nearest', **sel_dict),coord=bias_var.lon)
    plotting_map_m4 = add_cyclic_point(
        weighted_monthly_mean(merra_var[var]).sel(method='nearest', **sel_dict))

    xticks = np.arange(-180, 181, 60)
    yticks = np.arange(-60, 61, 30)

    # Plotting
    colormap = plt.cm.BrBG if var == 'PRECT_MERRA' else plt.cm.bwr
    e = ax.contourf(plot_lons, bias_var.lat, plotting_map_c4, levels=clev, cmap=colormap, extend='both', transform=ccrs.PlateCarree())
    ax.contour(plot_lons, bias_var.lat, plotting_map_m4, colors='grey', transform=ccrs.PlateCarree())

    # Map features and formatting (unrolled from the previous loop)
    ax.coastlines()
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(labelsize=12)

    # Adjusted colorbar position to align with a single panel
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(e, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(clab)

    plot_title = f'{case_name} {var}' if levsel is None else f'{case_name} {var} at {levsel} hPa'
    plt.suptitle(plot_title, y=0.88)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_precipitation_bias(pr_diff, model_interp, case_name, output_path):
    """Maps dynamic horizontal rainfall distributions against observations for a single file (2x1 layout)."""
    # --- Custom Colormap Setup ---
    cmap = plt.cm.BrBG
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = cmap(0)
    cmaplist[cmap.N-1] = cmap(0.99)
    for ii in range(116, 137):
        cmaplist[ii] = [1, 1, 1, 1]
    cmap_custom = cmap.from_list('Custom_BrBG', cmaplist, cmap.N)
    
    # --- Create a 2x1 Subplot Layout ---
    # Adjusted figsize to be taller (8, 8) to accommodate two stacked maps + colorbars nicely
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    
    # --- Data Processing ---
    # Top Panel: pr_c_diff (Bias Map)
    plotting_map_bias, plot_lons = add_cyclic_point(weighted_monthly_mean(pr_diff), coord=pr_diff.lon)
    
    # Bottom Panel: spcam_c_interp (Absolute Intensity Map)
    plotting_map_abs = add_cyclic_point(weighted_monthly_mean(model_interp['PRECT']))

    # --- Plotting Ranges ---
    clevels = np.arange(-5, 5.2, .2)  # For bias
    clevpn = np.arange(0, 18, 2)       # For absolute intensity
    xticks = np.arange(-180, 181, 60)
    yticks = np.arange(-60, 61, 30)

    # --- Top Panel (axs[0]): Model Bias ---
    e = axs[0].contourf(plot_lons, pr_diff.lat, plotting_map_bias, levels=clevels, cmap=cmap_custom, extend='both', transform=ccrs.PlateCarree())
    cbar0 = fig.colorbar(e, ax=axs[0], orientation='horizontal', pad=0.12, shrink=0.7)
    cbar0.ax.tick_params(labelsize=10)
    cbar0.set_label('mm/day')

    # --- Bottom Panel (axs[1]): Absolute Intensity ---
    a = axs[1].contourf(plot_lons, pr_diff.lat, plotting_map_abs, levels=clevpn, cmap='Blues', extend='both', transform=ccrs.PlateCarree())
    cbar1 = fig.colorbar(a, ax=axs[1], orientation='horizontal', pad=0.12, shrink=0.7)
    cbar1.ax.tick_params(labelsize=10)
    cbar1.set_label('mm/day')

    # --- Geographic Features & Gridlines ---
    for ax in axs:
        ax.coastlines()
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        ax.tick_params(labelsize=10)

    mean_pr_bias = np.round(weighted_monthly_mean(pr_diff).mean().values,3)
    plt.suptitle(f'{case_name} precip: global bias {mean_pr_bias} mm/day', y=0.92)

    # --- Save and Close ---
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_normalized_error_profiles(case_ds, control_ds, merra_ds, case_name, output_path):
    """Plots multi-variable vertical cross-sections of normalized profile error metrics"""
    
    control_rmse = compute_weighted_rmse_season(merra_ds, control_ds)
    case_rmse = compute_weighted_rmse_season(merra_ds, case_ds)

    normalized_rmse = case_rmse / control_rmse

    has_lev = 'lev' in normalized_rmse.dims
    vert_dim = 'lev' if has_lev else 'plev'
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    variables = ['U', 'V', 'T', 'Q']
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple']

    for i, var in enumerate(variables):
        ax.plot(normalized_rmse[var], normalized_rmse[vert_dim], color=colors[i], label = var)

    ax.vlines(1, 0, 1000, 'grey','dashdot')
    ax.set_ylabel('hPa')
    ax.set_xlabel('Avg Normalized Error')
    ax.set_xlim([0, 1.2])
    ax.grid(True, linestyle=':', alpha=0.6)

    # Invert the vertical dimension on shared-axis coordinates to mimic normal atmospheric pressure profiles
    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.suptitle(f'{case_name}', y=0.95)

    # --- Save and Close ---
    fig.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    normalized_rmse.to_netcdf(output_path.with_suffix('.nc'),mode='w')


def plot_horizontal_tendency_matrix(tend_season, case_name, output_path):
    """Constructs a 4x4 matrix mapping model tendencies across pressure coordinates."""

    levels = [100, 300, 800, 1000]
    all_var_names = ['UDIFF', 'VDIFF', 'SDIFF', 'QDIFF']
    all_labels = ["m/s", "m/s", "J/kg", "g/kg"]

    clevu = np.linspace(-1.25,1.25,25)
    clevv = np.linspace(-.75,.75,25)
    clevq = np.linspace(-5e-4,5e-4,25)
    clevs = np.linspace(-1000,1000,25)
    all_clevels = [clevu, clevv, clevs, clevq]

    # Dynamically extract dimensions using standard coordinates
    sample_var = tend_season[all_var_names[0]]
    has_lev = 'lev' in sample_var.dims
    vert_dim = 'lev' if has_lev else 'plev'

    # Decide whether to include SDIFF
    sdiff_is_zero = np.all(tend_season['SDIFF'].values == 0)

    if sdiff_is_zero:
        keep_vars = [v for v in all_var_names if v != 'SDIFF']
    else:
        keep_vars = all_var_names

    keep_indices = [all_var_names.index(v) for v in keep_vars]

    var_names = [all_var_names[idx] for idx in keep_indices]
    clevels = [all_clevels[idx] for idx in keep_indices]
    labels = [all_labels[idx] for idx in keep_indices]

    n_rows = len(levels)
    n_cols = len(var_names)

    plotting_maps = np.empty([n_rows, n_cols, len(tend_season.lat), len(tend_season.lon) + 1])
    for i, lev in enumerate(levels):
        for j, var_name in enumerate(var_names):
            sel_dict = {vert_dim: levels[i]}
            data_slice = weighted_monthly_mean(tend_season[var_names[j]]).sel(method='nearest', **sel_dict)
            plotting_maps[i, j, :, :], plot_lons = add_cyclic_point(data_slice, coord=tend_season.lon)

    # Adjust figure width based on number of columns
    fig_width = 14 if n_cols == 4 else 10.5

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, 9), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}, sharex=True, sharey=True)
    xticks = np.arange(-180, 181, 90)
    yticks = np.arange(-60, 61, 60)
    axes_obj = []

    for i in range(n_rows):
        for j in range(n_cols):
            pl = axs[i,j].contourf(plot_lons, tend_season.lat, plotting_maps[i,j,:,:], levels=clevels[j], cmap=plt.cm.bwr, extend='both', transform=ccrs.PlateCarree())
            axs[i,j].set_title(f'{var_names[j]}{levels[i]}', fontsize=12)
            if i == 0:
                axes_obj.append(pl)

    for ax in axs.flatten():
        ax.coastlines()
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        ax.tick_params(labelsize=10)

    for idx in range(n_cols):
        ax_pos = axs[-1, idx].get_position()
        cb_ax = fig.add_axes([ax_pos.x0,0.06,ax_pos.width,0.02])
        cb = fig.colorbar(axes_obj[idx],cax=cb_ax,orientation="horizontal")
        if var_names[idx] == "QDIFF":
            cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 1000:g}"))
        cb.set_label(labels[idx])
        cb.ax.tick_params(labelsize=12, rotation=45)

    fig.suptitle(f'{case_name} Tendencies', y=0.92, fontsize=14)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_vertical_tendency_profiles(tend_season, case_name, output_path):
    """Maps continuous zonal profiles of localized model tendency vectors."""
    
    tend_latave = take_spatial_average_xr(tend_season, orientation='lon')

    fig, ax = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    
    clevu = np.linspace(-.8,.8,25)
    clevv = np.linspace(-.3,.3,25)
    clevq = np.linspace(-1e-4,1e-4,25)
    clevs = np.linspace(-500,500,25)

    has_lev = 'lev' in tend_latave.dims
    vert_coord = tend_latave.lev if has_lev else tend_latave.plev

    # U-Tendency
    e = ax[0,0].contourf(tend_latave.lat, vert_coord, weighted_monthly_mean(tend_latave.UDIFF),
                         levels=clevu, extend='both', cmap=plt.cm.bwr, norm=mcolors.TwoSlopeNorm(0))
    fig.colorbar(e, ax=ax[0,0], label='m/s')
    ax[0,0].set_title('UDIFF')

    # V-Tendency
    f = ax[0,1].contourf(tend_latave.lat, vert_coord, weighted_monthly_mean(tend_latave.VDIFF),
                         levels=clevv, extend='both', cmap=plt.cm.bwr, norm=mcolors.TwoSlopeNorm(0))
    fig.colorbar(f, ax=ax[0,1], label='m/s')
    ax[0,1].set_title('VDIFF')

    # Q-Tendency
    g = ax[1,0].contourf(tend_latave.lat, vert_coord, weighted_monthly_mean(tend_latave.QDIFF),
                         levels=clevq, extend='both', cmap=plt.cm.bwr, norm=mcolors.TwoSlopeNorm(0))
    fig.colorbar(g, ax=ax[1,0], label='kg/kg')
    ax[1,0].set_title('QDIFF')

    # S-Tendency (Thermodynamic Profile Transformation)
    if np.all(tend_latave.SDIFF == 0):
        fig.delaxes(ax[1,1])
    else:
        h = ax[1,1].contourf(tend_latave.lat, vert_coord, weighted_monthly_mean(tend_latave.SDIFF),
                            levels=clevs, extend='both', cmap=plt.cm.bwr, norm=mcolors.TwoSlopeNorm(0))
        fig.colorbar(h, ax=ax[1,1], label='K')
        ax[1,1].set_title('SDIFF')

    ax[0,0].invert_yaxis()
    ax[0,0].set_ylabel('hPa')
    ax[1,0].set_ylabel('hPa')
    ax[1,0].set_xlabel('Latitude')
    ax[1,1].set_xlabel('Latitude')

    fig.suptitle(f'{case_name} Tendencies', y=.95)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_transient_eddies(ds_transients, ds_obs_transients, case_name, output_path):
    """Evaluates dynamic variance covariance envelopes across distinct seasons."""
    modl_vars = ['UpUp', 'VpVp', 'UpVp', 'EKE', 'VpQp']
    obs = ['UpUp', 'VpVp', 'UpVp', 'EKE', 'VpQp']

    has_lev = 'lev' in ds_transients.dims
    vert_dim = 'lev' if has_lev else 'plev'
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(25, 5), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    ax = ax.ravel()

    for nummy in range(5):
        Vary = modl_vars[nummy]
        
        # Extract standard vertical surface level constraints per-variable type
        levdo = 850 if Vary == 'VpQp' else 200
        sel_dict = {vert_dim: levdo}
        plotter_scaling = 1000 if Vary == 'VpQp' else 1
        
        # Dynamic Divergent Boundary Norm Conversions
        if Vary == 'VpQp':
            clevels = np.arange(-5.5, 6, .5)
            cmap_base = plt.cm.BrBG
        elif Vary == 'UpVp':
            clevels = np.arange(-50,55,5)
            cmap_base = plt.cm.RdYlBu_r 
        else:
            clevels = np.arange(-100, 110, 10)
            cmap_base = plt.cm.RdYlBu_r

        cmaplist = [cmap_base(i) for i in range(cmap_base.N)]
        cmaplist[0] = cmap_base(0)
        cmaplist[cmap_base.N-1] = cmap_base(0.99)
        
        # Standardize white zero-crossing window coordinates across variables
        mid_start, mid_end = (115, 137) if Vary == 'VpQp' else (120, 136)
        for ii in range(mid_start, mid_end):
            cmaplist[ii] = [1, 1, 1, 1]
        cmap_custom = mcolors.LinearSegmentedColormap.from_list('Custom_Eddy', cmaplist, cmap_base.N)
        norm = mcolors.BoundaryNorm(clevels, cmap_custom.N)

        # Isolate structures across specified surfaces
        mod_slice = weighted_monthly_mean(ds_transients).sel(method='nearest', **sel_dict)[Vary]
        obs_slice = weighted_monthly_mean(ds_obs_transients).sel(method='nearest', **sel_dict)[Vary]
        plot_data = (mod_slice - obs_slice).squeeze() * plotter_scaling

        cyclic_data, plot_lons = add_cyclic_point(plot_data, coord=ds_transients.lon)

        im = ax[nummy].contourf(plot_lons, ds_transients.lat, cyclic_data, levels=clevels, cmap=cmap_custom, norm=norm, extend='both', transform=ccrs.PlateCarree())
        ax[nummy].coastlines()
        ax[nummy].set_title(f'Δ {Vary} at {levdo} hPa', fontsize=14)
        
        cb = fig.colorbar(im, ax=ax[nummy], orientation='horizontal', pad=0.05, shrink=0.7)
        #cb.ax.tick_params(labelsize=8)

    fig.suptitle(f'Transients {case_name}', y=.75, fontsize=16)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def run_and_save_rmse_analysis(model_ds, control_ds, obs_ds, table_configs, months, time_sel, lf_scale,
                               case_name, output_path):
    """
    Computes spatial actual/bootstrapped RMSE metrics over multiple regions,
    saves as xlsx file, and plots bar plot.
    """
    
    regions = ['Global', 'Tropics', 'Extratropics', 'Land', 'Ocean']

    # open temp LANDFRAC file
    lf_map = xr.open_dataset('/n/holystore01/INTERNAL_REPOS/CLIMATE_MODELS/cesm_2_1/inputdata/atm/cam/topo/USGS-gtopo30_1.9x2.5_remap_c050602.nc')['LANDFRAC']
    lat = model_ds.lat

    records = [] 
    for vardo, config in table_configs.items():

        da_sample = model_ds[vardo]
        lev_dim = None
        if 'plev' in da_sample.dims:
            lev_dim = 'plev'
        elif 'lev' in da_sample.dims:
            lev_dim = 'lev'
            
        levels_to_process = config.get('levels', [None]) if lev_dim else [None]

        for levsel in levels_to_process:
            
            # Create a label for the table row (e.g., 'T' becomes 'T850', or stays 'T' if 2D)
            row_var_name = f"{vardo}{levsel}" if levsel is not None else vardo
            
            # Slice the dataset at the current level if applicable
            if lev_dim and levsel is not None:
                # method='nearest' safely handles minor floating point discrepancies in coordinate definitions
                obs_sub = obs_ds[vardo].sel(**{lev_dim: levsel}, method='nearest')
                model_sub = model_ds[vardo].sel(**{lev_dim: levsel}, method='nearest')
                control_sub = control_ds[vardo].sel(**{lev_dim: levsel}, method='nearest')
            else:
                if vardo == 'PRECT':
                    # precip dataset from ncar-ncep
                    ncep_pr = xr.open_dataset('~/holylfs06/ncep-ncar/prate/prate.day.mean.nc')['prate']
                    ncep_pr_season = ncep_pr.groupby('time.month').mean(dim='time').sel(month=months)
                    ncep_pr_season = ncep_pr_season * 86400.0 # kg/m2/s to mm/day
                    ncep_pr_interp = ncep_pr_season.interp(lat=model_ds.lat, lon=model_ds.lon, method='nearest') 
                    obs_sub = ncep_pr_interp.rename('PRECT')

                    # precip bias
                    model_sub = model_ds['PRECT'] * 86400.0 * 1000.0 # m/s to mm/day
                    control_sub = control_ds['PRECT'] * 86400.0 * 1000.0 # m/s to mm/day
                else:
                    obs_sub = obs_ds[vardo]
                    model_sub = model_ds[vardo]
                    control_sub = control_ds[vardo]

            # --- Compute Metrics across Regions & Seasons ---
            for area in regions:
                    
                # Apply Masking
                obs_m = apply_regional_mask(obs_sub, area.lower(), lat, lf_map, lf_scale)
                model_m = apply_regional_mask(model_sub, area.lower(), lat, lf_map, lf_scale)
                control_m = apply_regional_mask(control_sub, area.lower(), lat, lf_map, lf_scale)
                    
                actual_rmse_model = compute_weighted_rmse_season(obs_m, model_m).values
                actual_rmse_control = compute_weighted_rmse_season(obs_m, control_m).values
                
                # Compute Performance Improvement Percent (PI) as a string to match your custom parser
                if actual_rmse_control != 0:
                    pi_val = ((actual_rmse_control - actual_rmse_model) / actual_rmse_control) * 100
                else:
                    pi_val = 0.0
                pi_str = np.round(pi_val, 3)
                rmse_flt = np.round(actual_rmse_model, 3)
                control_flt = np.round(actual_rmse_control, 3) 

                records.append([case_name, time_sel, row_var_name, area, pi_str, rmse_flt, control_flt])
        
    export_df = pd.DataFrame(records, columns=['Case','Season', 'Var', 'Region', 'Percent Change', 'Model RMSE', 'Control RMSE'])
    export_df.to_excel(output_path.with_suffix('.xlsx'), index=False)

    # # Quick Call to plot right away
    plot_grouped_rmse(export_df, save_path=output_path)

def plot_grouped_rmse(df, save_path):
    # 2. Identify layout dimensions
    unique_vars = df['Var'].unique()  # ['T1000', 'T850', 'T500', 'T250']
    unique_areas = df['Region'].unique() 
    season = df['Season'].iloc[0]
    
    # Setup subplots - Compact sizing: width 8, and only 2.2 height per row
    fig, axs = plt.subplots(len(unique_vars), 1, figsize=(4, 2.2 * len(unique_vars)), sharex=True)
    if len(unique_vars) == 1:
        axs = [axs]
        
    # Bar grouping setup (centered at 0 since we are plotting 1 bar per region now)
    x_coords = np.arange(len(unique_areas))
    bar_width = 0.5  # slightly wider bar to fill out the compact layout nicely
    
    # 3. Build subplots dynamically
    for idx, vardo in enumerate(unique_vars):
        ax = axs[idx]
        var_subset = df[df['Var'] == vardo].set_index('Region').reindex(unique_areas).reset_index()
        
        # Pull values out as a float array
        y_values = var_subset['Percent Change'].astype(float)
        
        # Assign Green (#2ecc71) if >= 0, else Red (#e74c3c)
        bar_colors = ['#2ecc71' if val >= 0 else '#e74c3c' for val in y_values]
        
        # Plot single centered bar per region
        rects = ax.bar(x_coords, y_values, bar_width, 
                       color=bar_colors, edgecolor='black', alpha=0.85)
        
        # Add a crisp reference line at 0% change
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
        
        # Grid, framing, and labels
        ax.set_ylabel(f'{vardo}\n% Improv.', fontsize=10)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.tick_params(labelsize=10)
            
    # Apply global x-axis tags to the bottom subplot frame
    axs[-1].set_xticks(x_coords)
    axs[-1].set_xticklabels(unique_areas, fontsize=11)
    axs[-1].tick_params(axis='x', labelrotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight', dpi=250)
    plt.show()

# =====================================================================
# PART 3: AUTOMATED PIPELINE EXECUTION ENGINE
# =====================================================================
# Helper function to parse string booleans from Bash environment
def get_env_bool(name, default="False"):
    return os.getenv(name, default).lower() in ("true", "1", "yes")

if __name__ == "__main__":

    # 1. Parse Boolean Flags
    os.environ.get('test_var', 'default_value')
    do_bias_plots      = get_env_bool("DO_BIAS_PLOTS", "False")
    do_tendency_plots  = get_env_bool("DO_TENDENCY_PLOTS", "False")
    do_transient_plots = get_env_bool("DO_TRANSIENT_PLOTS", "False")
    REGRID             = get_env_bool("REGRID", "False")

    # 2. Parse Text / Case Variables
    CASE          = os.getenv("CASE")
    CONTROL       = os.getenv("CONTROL")
    TENDENCY_CASE = os.getenv("TENDENCY_CASE")
    TIME_LABEL    = os.getenv("TIME_LABEL")
    DO_SURFACE    = os.getenv("DO_SURFACE")
    DO_TOA        = os.getenv("DO_TOA")

    # 3. Parse and Unpack Time Arrays
    # Reads the comma-separated string from bash (e.g., "12,1,2") and maps it to a list of ints
    months_str = os.getenv("MONTHS")
    MONTHS     = [int(m) for m in months_str.split(",")]

    print(f"Case: {CASE} | Time Frame: {TIME_LABEL}")
    print(f"Months to process: {MONTHS}")
    print(f"Plotting configurations: Bias={do_bias_plots}, Tendency={do_tendency_plots}\n")

    # Years remain standard array definitions or can be conditioned based on flags
    ANALYSIS_YEARS  = np.arange(1980, 2010)
    TRANSIENT_YEARS = np.arange(1980, 1990)

    CASE_DIR = f'/n/home04/sweidman/holylfs06/CESM2_corrector_out/Run/archive/{CASE}/atm/hist/{CASE}'
    #CASE_DIR = f'/n/home04/sweidman/holylfs06/CESM2_corrector_out/Run/{CASE}/run/{CASE}'
    CONTROL_DIR = f'/n/home04/sweidman/holylfs06/CESM2_corrector_out/Run/archive/{CONTROL}/atm/hist/{CONTROL}'
    TENDENCY_DIR = f'/n/home04/sweidman/holylfs06/IC_CESM2/monave/{TENDENCY_CASE}'
    MERRA_DIR = '/n/home04/sweidman/holylfs06/MERRA2_OG/MERRA2_f19/'
    CLIMO = 'monave30' # for years in MERRA climatology 
    #MERRA_DIR = '/n/home04/sweidman/holylfs06/ERA5/ERA5_f19/'

    OUTPUT_IMAGE_DIR = Path(f'./diagnostic_outputs/{CASE}/{TIME_LABEL}')
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


    # -----------------------------------------------------------------
    # PIPELINE STEP 1: Compute State Variable Biases
    # -----------------------------------------------------------------

    if do_bias_plots:
        print(">>> Opening monthly files")
        model_ds, model_diff, model_interp, merra_ds, pr_diff = calculate_bias_ds(CASE_DIR, MERRA_DIR, ANALYSIS_YEARS, MONTHS, CLIMO, REGRID, DO_SURFACE, DO_TOA)

        print(">>> Plotting latxlev bias")
        plot_lat_vs_level_bias(model_diff, merra_ds, CASE, OUTPUT_IMAGE_DIR / f"{TIME_LABEL}_latxlev_bias.png")

        print(">>> Plotting lonxlat bias")
        plot_configs = {
            'T': {'clev': np.arange(-4,4.5,.5),
                'clab': '[K]','levels': [975, 850, 500, 250]},
            'Q': {'clev': np.arange(-1.4, 1.5, 0.2) / 1000,
                'clab': '[kg/kg]','levels': [975, 850]},
            'U': {'clev': np.arange(-4, 4.1, 0.4),
                'clab': '[m/s]','levels': [850, 250]},
            'V': {'clev': np.arange(-2, 2.1, 0.2),
                'clab': '[m/s]','levels': [850, 250]}}

        if DO_SURFACE:
            plot_configs |= {'SHFLX': {'clev': np.linspace(-50,50,25),
                                'clab': '[W/m^2]','levels': [None]},
                            'LHFLX': {'clev': np.linspace(-50,50,25),
                                'clab': '[W/m^2]','levels': [None]},
                            'PRECT_MERRA': {'clev': np.arange(-5, 5.2, .2),
                                'clab': '[mm/day]','levels': [None]},
                            'TS': {'clev': np.linspace(-10,10,25),
                                'clab': '[K]','levels': [None]},
                            'TREFHT': {'clev': np.linspace(-10,10,25),
                                'clab': '[K]','levels': [None]},
                            'TMQ': {'clev': np.linspace(-10,10,25),
                                'clab': '[kg m-2]','levels': [None]},
                            'OMEGA500': {'clev': np.linspace(-.05,.05,25),
                                'clab': '[Pa/s]','levels': [None]},
                            'Z500': {'clev': np.linspace(-50,50,25),
                                'clab': '[m]','levels': [None]}}
        if DO_TOA:
            plot_configs |= {'SWCF': {'clev':np.linspace(-50,50,25),
                                'clab': '[W/m^2]','levels': [None]},
                            'LWCF': {'clev':np.linspace(-30,30,25),
                                'clab': '[W/m^2]','levels': [None]},
                            'CLDTOT': {'clev': np.linspace(-.75,.75,25),
                                'clab': '[fraction]','levels': [None]}}
            

        for var, config in plot_configs.items():

            for levsel in config['levels']:
                outfile = f"{TIME_LABEL}_{var}.png" if levsel is None else f"{TIME_LABEL}_{var}{levsel}.png"

                plot_lat_vs_lon_bias(model_diff, merra_ds, CASE, 
                                     var, levsel, config['clev'], config['clab'], 
                                     OUTPUT_IMAGE_DIR / outfile)
                    

        print(">>> Plot precip bias")
        plot_precipitation_bias(pr_diff, model_interp, CASE, OUTPUT_IMAGE_DIR / f"{TIME_LABEL}_precip.png")

        if CASE != 'cam_control':
            print(">>> Plot normalized error")
            # load raw control and model
            control_ds = open_monthly_model(CONTROL_DIR, ANALYSIS_YEARS, MONTHS, regrid=REGRID)
            plot_normalized_error_profiles(model_ds, control_ds, merra_ds, CASE, OUTPUT_IMAGE_DIR / f"{TIME_LABEL}_norm_error")

            print(">>> Save RMSE excel")
            table_configs = {
                'T': {'levels': [975, 850, 500, 250]},
                'Q': {'levels': [975, 850]},
                'U': {'levels': [850, 250]},
                'V': {'levels': [850, 250]},
                'PRECT': {'levels': None}}
            if DO_SURFACE:
                control_sfc = open_monthly_model(CONTROL_DIR, ANALYSIS_YEARS, MONTHS, var_list=['TS','TMQ','TREFHT','PRECT','SHFLX','LHFLX'],regrid=False)
                control_sfc['PRECT_MERRA'] = control_sfc['PRECT'] * 86400.0 * 1000.0
                control_ds = xr.merge([control_ds, control_sfc], compat='minimal') 
                table_configs |= {'SHFLX': {'levels': None},
                                'LHFLX': {'levels': None},
                                'PRECT_MERRA': {'levels': None},
                                'TS': {'levels': None},
                                'TREFHT': {'levels': None},
                                'TMQ': {'levels': None}}
            if DO_TOA:
                control_toa = open_monthly_model(CONTROL_DIR, ANALYSIS_YEARS, MONTHS, var_list=['CLDTOT','SWCF','LWCF'],regrid=False)
                control_ds = xr.merge([control_ds, control_toa], compat='minimal') 
                table_configs |= {'CLDTOT': {'levels': None},
                                'SWCF': {'levels': None},
                                'LWCF': {'levels': None}}
            run_and_save_rmse_analysis(model_ds, control_ds, merra_ds, table_configs, MONTHS, TIME_LABEL, lf_scale=0.5,
                                    case_name = CASE, output_path = OUTPUT_IMAGE_DIR / f"{TIME_LABEL}_rmse_table")
        
    if do_tendency_plots:
        print(">>> Opening tendency")
        tendency_dataset = open_monthly_tendencies(TENDENCY_DIR, MONTHS, regrid=False)
        
        print(">>> Plotting lonxlat tendencies")
        plot_horizontal_tendency_matrix(tendency_dataset, CASE, OUTPUT_IMAGE_DIR / f"{TIME_LABEL}_lonxlat_tendencies.png")

        print(">>> Plotting latxlev tendencies")
        plot_vertical_tendency_profiles(tendency_dataset, CASE, OUTPUT_IMAGE_DIR / f"{TIME_LABEL}_latxlev_tendencies.png")

    if do_transient_plots:
        print(">>> Opening daily files")
        daily_model = open_daily_regex(CASE_DIR, 'model', TRANSIENT_YEARS, MONTHS, varsel = ['U','V','T','Q', 'PS'], regrid=REGRID)
        daily_obs = open_daily_regex(MERRA_DIR, 'MERRA', TRANSIENT_YEARS, MONTHS, varsel = ['U','V','T','Q', 'PS'], regrid=REGRID)

        transients_model = calculate_transients(daily_model)
        transients_obs = calculate_transients(daily_obs)
        # TODO: save the transients as a file in the case folder so they don't need to be recalculated 

        print(">>> Plotting transients")
        plot_transient_eddies(transients_model, transients_obs, case_name=CASE, output_path = OUTPUT_IMAGE_DIR / f"{TIME_LABEL}_transients.png")

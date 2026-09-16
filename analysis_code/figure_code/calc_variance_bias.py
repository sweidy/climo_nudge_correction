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

# External climate computation library
import geocat.comp as gcomp

def open_daily_regex(filedir, case_type, years, months, varsel=None, regrid=False):
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
        return regrid_plev(ds, target_plev=np.array([200, 500, 850]) * 100)
    else:
        return ds

def regrid_plev(ds, target_plev=np.array([1, 10, 50] + list(range(100, 901, 50)) + list(range(925, 1001, 25)), dtype=float)*100):
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
                    extrapolate=True,
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
                    extrapolate=True,
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
    return ds_pressure

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

    return transients

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

def transient_rmse_records(model, obs, case, season):
    """Calculates global RMSE for every transient and returns table rows."""
    records = []
    for var in model.data_vars:
        rmse = compute_weighted_rmse(obs[var], model[var])
        if rmse.dims:
            df = rmse.to_dataframe(name='RMSE').reset_index()
        else:
            df = pd.DataFrame({'RMSE':[rmse.item()]})
        df.insert(0, 'Transient', var)
        df.insert(0, 'Season', season)
        df.insert(0, 'Case', case)
        records.append(df)
    return pd.concat(records, ignore_index=True)

if __name__ == "__main__":

    #case_list = ['apply_nudge_to_taper_uvq','apply_nudge_to_taper_uvq_iter8','apply_nudge_param_coef_1',
    #             'apply_nudge_param_coef_0.5','apply_nudge_param_coef_0.12','apply_nudge_control_norun_uvq']
    case_list = ['apply_nudge30_coef_1']

    CONTROL = 'cam_control'
    CONTROL_DIR = f'/n/home04/sweidman/holylfs06/CESM2_corrector_out/Run/archive/{CONTROL}/atm/hist/{CONTROL}'
    MERRA_DIR = '/n/home04/sweidman/holylfs06/MERRA2_OG/MERRA2_f19/'
    TRANSIENT_YEARS = np.arange(1980, 2010)
    seasons = {'DJF':[12,1,2], 'MAM':[3,4,5], 'JJA':[6,7,8], 'SON':[9,10,11]}
    all_months = np.arange(1, 13)

    outpath = '/n/home04/sweidman/mean_state_corrector/corrector_scripts/diagnostic_outputs/'
    outfile = outpath + f'transient_global_rmse.{TRANSIENT_YEARS[0]}-{TRANSIENT_YEARS[-1]}.xlsx'

    if os.path.exists(outfile):
        rmse_df = pd.read_excel(outfile)
        rmse_tables = [rmse_df]
    else:
        rmse_df = pd.DataFrame()
        rmse_tables = []

    def case_is_complete(case):
        if rmse_df.empty:
            return False
        case_rows = rmse_df[rmse_df['Case'] == case]
        return set(case_rows['Season']) == set(seasons)

    def save_results():
        global rmse_df
        rmse_df = pd.concat(rmse_tables, ignore_index=True)
        rmse_df.to_excel(outfile, index=False)

    print('Calculating observations')
    transient_merra_output = MERRA_DIR + f'MERRA2.transients.{TRANSIENT_YEARS[0]}-{TRANSIENT_YEARS[-1]}.nc'
    if os.path.isfile(transient_merra_output):
        transients_obs = xr.open_dataset(transient_merra_output)
    else:
        daily_obs = open_daily_regex(MERRA_DIR, 'MERRA', TRANSIENT_YEARS, all_months, varsel=['U','V','Q','T','PS'], regrid=True)
        transients_obs = calculate_transients(daily_obs)
        transient_output = MERRA_DIR + f'MERRA2.transients.{TRANSIENT_YEARS[0]}-{TRANSIENT_YEARS[-1]}.nc'
        transients_obs.to_netcdf(transient_output)
        daily_obs.close()

    if case_is_complete(CONTROL):
        print(f'Skipping completed case: {CONTROL}')
    else:
        print('Calculating control')
        transient_control_output = CONTROL_DIR + f'.transients.{TRANSIENT_YEARS[0]}-{TRANSIENT_YEARS[-1]}.nc'
        if os.path.isfile(transient_control_output):
            transients_control = xr.open_dataset(transient_control_output)
        else:
            daily_control = open_daily_regex(CONTROL_DIR, 'model', TRANSIENT_YEARS, all_months, varsel=['U','V','Q','T','PS'], regrid=True)
            transients_control = calculate_transients(daily_control)
            transient_output = CONTROL_DIR + f'.transients.{TRANSIENT_YEARS[0]}-{TRANSIENT_YEARS[-1]}.nc'
            transients_control.to_netcdf(transient_output)
            daily_control.close()

        control_records = []
        for season, months in seasons.items():
            control_season = transients_control.sel(month=months)
            obs_season = transients_obs.sel(month=months)
            control_records.append(transient_rmse_records(control_season, obs_season, CONTROL, season))

        if not rmse_df.empty:
            rmse_df = rmse_df[rmse_df['Case'] != CONTROL]
            rmse_tables = [rmse_df]

        rmse_tables.extend(control_records)
        save_results()
        transients_control.close()

    for case in case_list:
        if case_is_complete(case):
            print(f'Skipping completed case: {case}')
            continue

        print(f'Calculating {case}')
        case_dir = f'/n/home04/sweidman/holylfs06/CESM2_corrector_out/Run/archive/{case}/atm/hist/iter7/{case}'
        transient_case_output = case_dir + f'.transients.{TRANSIENT_YEARS[0]}-{TRANSIENT_YEARS[-1]}.nc'
        if os.path.isfile(transient_case_output):
            transients_model = xr.open_dataset(transient_case_output)
        else:
            daily_model = open_daily_regex(case_dir, 'model', TRANSIENT_YEARS, all_months, varsel=['U','V','Q','T','PS'], regrid=True)
            transients_model = calculate_transients(daily_model)
            transient_output = case_dir + f'.transients.{TRANSIENT_YEARS[0]}-{TRANSIENT_YEARS[-1]}.nc'
            transients_model.to_netcdf(transient_output)
            daily_model.close()

        case_records = []
        for season, months in seasons.items():
            model_season = transients_model.sel(month=months)
            obs_season = transients_obs.sel(month=months)
            case_records.append(transient_rmse_records(model_season, obs_season, case, season))

        if not rmse_df.empty:
            rmse_df = rmse_df[rmse_df['Case'] != case]
            rmse_tables = [rmse_df]

        rmse_tables.extend(case_records)
        save_results()

        transients_model.close()

    transients_obs.close()

    outpath = '/n/home04/sweidman/mean_state_corrector/corrector_scripts/diagnostic_outputs/'
    rmse_df = pd.concat(rmse_tables, ignore_index=True)
    rmse_df.to_excel(outpath+'transient_global_rmse.xlsx', index=False)
    

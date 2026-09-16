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

def open_all_monthly_model(path_str, years=np.arange(1980,2010), mons=np.arange(1,13), var_list=['U','V','T','Q','PS'], regrid=True):
    # Read in monthly model data and return as single dataset

    run_list = []
    print(path_str)

    for y in years:
        for m in mons:
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

    run_ds = xr.concat(run_list, dim='time')

    if 'PRECC' in run_ds.data_vars:
        run_ds['PRECT'] = run_ds['PRECC'] + run_ds['PRECL'] 
        
    if regrid:
        print('regrid model to plev')
        return regrid_plev(run_ds)
    else:
        return run_ds

def open_all_monthly_merra(path_str, years=np.arange(1980,2010), mons=np.arange(1,13), var_list=['U','V','T','Q','PS'], 
                           time_list=None, regrid=True):
    
    run_list = []

    if time_list is None:
        time_list = pd.date_range(start=f'{years[0]}-01', end=f'{years[-1]}-12', freq='MS')

    for y in years:
        for m in mons:
            path2 = f'months/MERRA2_{y:04}{m:02}.nc'
            ds = xr.open_dataset(path_str + path2, decode_times=False)
            run_list.append(ds[var_list])
    
    mon_concat = xr.concat(run_list, dim='time')
    mon_concat['time'] = time_list
        
    if regrid:
        print('regrid model to plev')
        return regrid_plev(mon_concat)
    else:
        return mon_concat

def bootstrap_normalized_rmse(merra_ds, control_ds, corrected_ds, n_boot=1000, seed=15):
    rng = np.random.default_rng(seed)

    years = np.arange(1980,2010)

    bootstrap_rmse = []

    for i in range(n_boot):

        if i%5 == 0:
            print(i)

        sampled_years = rng.choice(years, size=len(years), replace=True)

        # Concatenating year blocks retains repeated years in the bootstrap sample.
        control_sample = xr.concat([control_ds.sel(time=control_ds.time.dt.year == year) for year in sampled_years],dim="time")
        corrected_sample = xr.concat([corrected_ds.sel(time=corrected_ds.time.dt.year == year) for year in sampled_years],dim="time")
        merra_sample = xr.concat([merra_ds.sel(time=merra_ds.time.dt.year == year) for year in sampled_years],dim="time")

        control_mons = control_sample.groupby("time.month").mean()
        corrected_mons = corrected_sample.groupby("time.month").mean()
        merra_mons = merra_sample.groupby("time.month").mean()

        control_rmse = compute_weighted_rmse_season(merra_mons, control_mons)
        corrected_rmse = compute_weighted_rmse_season(merra_mons, corrected_mons)

        normalized_rmse = (corrected_rmse / control_rmse).isel(plev=slice(1, 27)).compute()
        bootstrap_rmse.append(normalized_rmse)

        del control_sample, corrected_sample, merra_sample
        del control_mons, corrected_mons, merra_mons
        del control_rmse, corrected_rmse, normalized_rmse

    return xr.concat(bootstrap_rmse, dim=xr.IndexVariable("bootstrap", np.arange(n_boot)))

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

def save_norm_bs(bs, outpath):

    bs_ds = xr.Dataset()

    for var in bs.data_vars:
        bs_ds[f'{var}_median'] = bs[var].quantile(0.50, dim='bootstrap', skipna=True).drop_vars('quantile')
        bs_ds[f'{var}_q5'] = bs[var].quantile(0.05, dim='bootstrap', skipna=True).drop_vars('quantile')
        bs_ds[f'{var}_q95'] = bs[var].quantile(0.95, dim='bootstrap', skipna=True).drop_vars('quantile')

    print(bs_ds)
    print(bs_ds.U_q5.values)
    bs_ds.to_netcdf(outpath,mode='w')

if __name__ == "__main__":

    control_case = 'cam_control'
    cases_bs = ['apply_nudge30_integrate_false']#, 'apply_nudge30_integrate_false', 'apply_nudge_control_norun_uvq']
    iteration = 'iter8'

    outpath = '/n/home04/sweidman/mean_state_corrector/figs_corrector_paper/'
    control_dir = f'/n/home04/sweidman/holylfs06/CESM2_corrector_out/Run/archive/{control_case}/atm/hist/{control_case}'
    merra_dir = '/n/home04/sweidman/holylfs06/MERRA2_OG/MERRA2_f19/'; climo="monave30"

    analysis_years = np.arange(1980,2010);  
    regrid=True

    months=[12,1,2]; time_period='DJF'
    for case in cases_bs:

        if os.path.isfile(outpath+f'{case}_{time_period}.nc'):
            continue
        else:
            control_ds = open_all_monthly_model(control_dir, years=analysis_years, mons=months, var_list=['U','V','T','Q','PS'], regrid=regrid)
            merra_ds = open_all_monthly_merra(merra_dir, mons=months, time_list = control_ds.time, regrid=regrid)
            try: 
                case_dir = f'/n/home04/sweidman/holylfs06/CESM2_corrector_out/Run/archive/{case}/atm/hist/{iteration}/{case}'
                case_ds = open_all_monthly_model(case_dir, years=analysis_years, mons=months, var_list=['U','V','T','Q','PS'], regrid=regrid)
            except FileNotFoundError:
                print('no iteration '+iteration)
                case_dir = f'/n/home04/sweidman/holylfs06/CESM2_corrector_out/Run/archive/{case}/atm/hist/{case}'
                case_ds = open_all_monthly_model(case_dir, years=analysis_years, mons=months, var_list=['U','V','T','Q','PS'], regrid=regrid)

            case_bs = bootstrap_normalized_rmse(merra_ds, control_ds, case_ds, n_boot=500)
            save_norm_bs(case_bs, outpath+f'{case}_{time_period}.nc')
            case_bs.close()

    months=[6,7,8]; time_period='JJA'
    for case in cases_bs:
        
        if os.path.isfile(outpath+f'{case}_{time_period}.nc'):
            continue
        else:
            control_ds = open_all_monthly_model(control_dir, years=analysis_years, mons=months, var_list=['U','V','T','Q','PS'], regrid=regrid)
            merra_ds = open_all_monthly_merra(merra_dir, mons=months, time_list = control_ds.time, regrid=regrid)
            try: 
                case_dir = f'/n/home04/sweidman/holylfs06/CESM2_corrector_out/Run/archive/{case}/atm/hist/{iteration}/{case}'
                case_ds = open_all_monthly_model(case_dir, years=analysis_years, mons=months, var_list=['U','V','T','Q','PS'], regrid=regrid)
            except FileNotFoundError:
                print('no iteration '+iteration)
                case_dir = f'/n/home04/sweidman/holylfs06/CESM2_corrector_out/Run/archive/{case}/atm/hist/{case}'
                case_ds = open_all_monthly_model(case_dir, years=analysis_years, mons=months, var_list=['U','V','T','Q','PS'], regrid=regrid)

            case_bs = bootstrap_normalized_rmse(merra_ds, control_ds, case_ds, n_boot=500)
            save_norm_bs(case_bs, outpath+f'{case}_{time_period}.nc')
            case_bs.close() 

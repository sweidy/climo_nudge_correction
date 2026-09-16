# functions primarily for analyizing principle components
# create phase diagrams
# split into distinct MJO events / active MJO days

import numpy as np
import pandas as pd
import xarray as xr

from scipy.signal import detrend

import mjoanalyses.general_mjo_tools as tools
import wavenumber_frequency_functions as wf
import mjoindices.omi.quick_temporal_filter as qtf
import mjoindices.omi.wheeler_kiladis_mjo_filter as wkf

def date_indices(pcs: pd.DataFrame, start_date: str, end_date: str) -> np.ndarray:
    """
    Find indices in dataframe that correspond to the days between two dates

    :param pcs: dataframe containing PCs and corresponding 'Date' coordinate
    :param start_date: string in form YYYY-MM-DD
    :param end_date: string in form YYYY-MM-DD

    :returns: np.array of indices corresponding to dates in dataframe
    """
    
    start_idx = np.where(pcs['Date'] == start_date)
    end_idx = np.where(pcs['Date'] == end_date)
    
    return np.arange(start_idx[0], end_idx[0])

def find_consecutive_events(pcs:pd.DataFrame, threshold:float, no_leap:bool = False):
    """
    Group MJO events by neighboring days. Label MJO event by number and tag event by length in days. 
    Add event criteria to PC dataframe
    
    :param pcs: dataframe of PCs with amplitude and date information
    :param threshold: scalar threshold for amplitude for event is active (>= threshold) or inactive (< threshold)
    :param no_leap: If True, assumes 365 days in a year

    :returns: dataframe with MJO event information as new columns
    """
    
    pcs_above_threshold = tools.restrict_data_above_threshold(pcs, threshold)
    
    # create series of number of days between neighboring days above threshold
    vals = [tools.ndays_btwn_dates(a,b, no_leap) for (a, b) in zip(pcs_above_threshold.Date.shift(), pcs_above_threshold.Date)]
    
    # increment event number whenever number of days between neighboring active events > 1
    count = 0
    result = []
    for r in vals:
        if r != 1:
            count += 1
        result.append(count)
    
    pcs_above_threshold['event_n'] = result
    
    # add label of length of event for active events
    event_lens = [len(pcs_above_threshold[pcs_above_threshold['event_n'] == n]) for n in range(1,count+1)]
    pcs_above_threshold['len_event'] = pcs_above_threshold.apply(lambda row: event_lens[row.event_n-1], axis=1)
        
    # fill in NaN for events below threshold
    pcs_full = pd.concat([pcs, pcs_above_threshold['event_n'], pcs_above_threshold['len_event']], axis=1)
    
    return pcs_full

def filter_arcodia(data: xr.DataArray) -> xr.DataArray:

    # wrong at this point
    
    #xmean = data.mean(dim='time')
    #xdetr = data - xmean
    xdetr = data

    running_mean = np.empty(xdetr.shape)
    # take the centered 120-day mean
    for idx_lat in range(0, xdetr.shape[1]):
        for idx_lon in range(0, xdetr.shape[2]):

            temp = np.squeeze(xdetr[:, idx_lat, idx_lon])
            tapered_data = wkf.taper_vector_to_zero(temp, 10)
            running_mean[:,idx_lat, idx_lon] = np.convolve(tapered_data, np.ones(120)/120, mode='same')

    #return running_mean
    return xdetr - running_mean


def remove_seasonal_cycle(data: xr.DataArray) -> xr.DataArray:
    """
    Detrend data, then run through 20-100-day bandpass filter to remove seasonal and 
    interannual variability.

    :param: data array with at least a time and latitude dimension, longitude optional. 

    :return: data array with linear trend removed, and put through a band-pass filter. 
    """

    xdetr = detrend(data.values, axis=0, type='linear') 

    filtered_data = np.empty(xdetr.shape)
    time_spacing = 1 # in days, TODO: read this from the data itself
    period_min = 20
    period_max = 100

    for idx_lat in range(0, xdetr.shape[1]):
        for idx_lon in range(0, xdetr.shape[2]):
            temp = np.squeeze(xdetr[:, idx_lat, idx_lon])
            #TODO: maybe use the full filter, rather than just the quick filter
            filtered_data[:, idx_lat, idx_lon] = qtf._perform_spectral_smoothing(temp, time_spacing, period_min, period_max)
    #xdetr = xr.DataArray(xdetr, dims=data.dims, coords=data.coords)
    #data = wf.rmvAnnualCycle(xdetr, 1, 1/100) # this is not the right bandpass filter I think
    
    data_out = xr.DataArray(filtered_data, dims=data.dims, coords=data.coords)
    return data_out

def average_over_phase(data: xr.DataArray) -> xr.DataArray:
    """
    Take the average of some variable, split by MJO phase.

    :param data: data array of any variable with dimensions time, lat, lon; data must include
    phase of MJO for each date. Will use the phase names / numbers as defined in the dataset. 

    :return: data array of variable split and averaged over MJO phase 
    """

    phase_list = np.unique(data.Phase).astype(int)
    
    mean_data_per_phase = np.empty((len(phase_list), len(data.lat), len(data.lon)))
    for idx, p in enumerate(phase_list):
        
        mean_data_per_phase[idx,:,:] = data[data.Phase == p].mean(dim="time")
        #print(p, sum(data.Phase == p))
        
    return xr.DataArray(data=mean_data_per_phase, dims=["Phase", "lat", "lon"],
                        coords=dict(
                        lat=data.lat,
                        lon=data.lon,
                        Phase=phase_list))

def combine_into_four_phases(data: xr.DataArray) -> xr.DataArray:
    """
    Split into standard 8 phases of MJO into 2-phase chunks. Combine phases 2&3, 4&5,
    6&7, and 8&1. New phases are labelled 2,4,6,8 in dataset. 

    :param data: data array of variable with dimensions time, lat, lon. data must have
    been split and averaged over MJO phase (8 phases) by :func:'average_over_phase'

    :return: data array of variable split and averaged over 4 MJO phases
    """
    
    phase_list = np.array([(2,3),(4,5),(6,7),(8,1)])
    phase_names = np.array([2,4,6,8])
    
    mean_data_per_phase = np.empty((len(phase_names), len(data.lat), len(data.lon)))
    for idx, ps in enumerate(phase_list):
        
        mean_data_per_phase[idx,:,:] = data.loc[[ps[0],ps[1]]].mean(dim="Phase")
        
    return xr.DataArray(data=mean_data_per_phase, dims=["Phase", "lat", "lon"],
                        coords=dict(
                        lat=data.lat,
                        lon=data.lon,
                        Phase=phase_names))

def convert_ms_to_mmday(data):
    # convert precipitation rate data in m/s to mm/day
    
    return data*86400*1000

def convert_kgm2s_to_mmday(data):
    # convert precipitation rate data in kg/m^2/s to mm/day

    return data*86400

def convert_m_to_gz(data):
    # convert raw height to geopotential height

    return data*9.81

def load_multiyear_lev_xr(filename_head, filename_tail, years, varname=None, lev=None, decode_times=True):
    # load in data from dataset where files are split by year
    # if multiple levels, extract one level (specified by level in same units as datset)

    datasets = []

    for y in years:

        full_filename = filename_head + str(y) + filename_tail
        data_oneyear = tools.load_data_xr(full_filename, varname, lev, decode_times)
        datasets.append(data_oneyear)

    return xr.concat(datasets, dim='time')


def process_variable_into_phases(data, pcs, var, real_dates, 
                                 interpolate_to_orig_grid=True, interp_lats=None, interp_lons=None, bounds_error=True,
                                 restrict_winter=False, mons=None,
                                 restrict_strong=False, threshold=None,
                                 filter_type='bandpass',
                                 combine_to_four_phases=False,
                                 timeBounds=None):
    
    """
    Put data into combined phase anomaly data array
    """
    
    if interpolate_to_orig_grid:
        if interp_lats is None:
            interp_lats = np.arange(90, -90.1, -2.5)
        if interp_lons is None:
            interp_lons = np.arange(0., 359.9, 2.5)
        data = tools.interpolate_spacial_grid_xr(data, interp_lats, interp_lons, bounds_error)
    
    var_pc = tools.add_pd_events_to_xr(data, pcs, real_dates=real_dates)
    
    if timeBounds is not None:
        var_pc = tools.restrict_time_xr(var_pc, timeBounds)
        
    if var == 'precip':
        if real_dates:
            var_pc = convert_kgm2s_to_mmday(var_pc)
        else:
            var_pc = convert_ms_to_mmday(var_pc)
    #elif var == 'gz':
    #    var_pc = convert_m_to_gz(var_pc)
            
    # take anomaly
    if filter_type == 'arcodia':
        var_anom = filter_arcodia(var_pc)
    elif filter_type == 'bandpass':
        var_anom = remove_seasonal_cycle(var_pc)
    var_anom = tools.split_time_into_components_xr(var_anom)
    
    # restrict data
    if restrict_winter:
        var_anom = tools.restrict_data_to_winter(var_anom, mons=None)
    if restrict_strong:
        if threshold is None: # default to 1 threshold
            threshold = 1
        var_anom = tools.restrict_data_above_threshold(var_anom, threshold)
    
    var_phase = average_over_phase(var_anom)
    if combine_to_four_phases:
        var_phase = combine_into_four_phases(var_phase)
        
    return var_phase 
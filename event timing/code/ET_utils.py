# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:40:14 2013

@author: Y
"""
import pandas as pd
import numpy as np
import os
import sys
import ConfigParser
import matplotlib.pylab as plt
#from pandas import datetime
#from datetime import time, date
#import datetime

config = ConfigParser.RawConfigParser()
config.read(os.path.abspath('E:\github\SystematicStrategies\idpresearch.cfg'))
sys.path.append(os.path.abspath(config.get('IDPResearch','libpath_windows')))

import idpresearch.siglib as siglib
sig = siglib.Sig()


def gen_weights(a,N): 
    """
    generate exponentially-weighted weights 
    a : initial weight
    N : number of weights    
    """
    ws = list()
    for i in range(N):
        w = a * ((1-a)**i)
        ws.append(w)
    return ws

def get_data_by_date(df, dates, date_i, col = ['Close']):
#   df --- dataframe strore all intraday data
#   dates --- all trading dates
#   date_i --- get data for that day  
    return df[dates[dates.index(date_i)].strftime('%Y-%m-%d')]
    
    
def get_hit_ratio(pl):
    df_temp = pd.DataFrame(pl)
    df_temp.fillna(value = 0, inplace= True)
    winning_events_no = len(df_temp[df_temp.cumsum(axis= 1).iloc[:, -1]>0])
    total_events_no = len(df_temp[df_temp.abs().cumsum(axis= 1).iloc[:, -1]>0])
    hit_ratio =  float(winning_events_no)/total_events_no
    return hit_ratio

def pl_analysis_by_section(df_pl, (st, ed)):
    #note in the section analysis, both end are included
    df_pl = df_pl.fillna(value = 0)
    pl_temp = df_pl.ix[ :, st:ed]
    #get rid of all zeros event
    pl_temp = pl_temp[(pl_temp.abs().cumsum(axis= 1).iloc[:, -1]>0).values]
    winning_events_no = len(pl_temp[(pl_temp.cumsum(axis= 1).iloc[:, -1]>0).values])
    total_events_no = len(pl_temp)
    pl_event = pl_temp.cumsum(axis= 1).iloc[:, -1]
    hit_ratio = float(winning_events_no)/total_events_no
    sharpe_ratio = pl_event.mean()/pl_event.std()
    kelly_ratio =  pl_event.mean()/pl_event.var()    
    return hit_ratio, sharpe_ratio, kelly_ratio
    
def pl_analysis_by_section_dict(df_pl, section_dict):
    hit_ratio_dict = {}
    sharpe_ratio_dict = {}
    kelly_ratio_dict = {}
    for key,value in section_dict.items():
        hit_ratio_dict[key], sharpe_ratio_dict[key], kelly_ratio_dict[key] = pl_analysis_by_section(df_pl, value)
    return hit_ratio_dict, sharpe_ratio_dict, kelly_ratio_dict 

    
def calc_gaussian_kernel_regression_weighted(X, Y, WEIGHT, x, bw = 0.05):
    """
    Calculate kernel regression using the gaussian kernel (regression type is local constant estimator).

    Parameters
    ----------
    X : list or array
        Independent variable
    Y : list or array
        Dependent variable
    WEIGHT: List or array
        Weight for dependent variable    
    x : list or array
        Independent points where to evaluate the regression
    bw : float, optional
        Bandwidth to be used to perform regression (defaults to 0.05)

    Returns
    -------
    f_x : list or array
        f(x), regression values evaluated at x
    var_f_x : list or array
        Variance of f(x)

    Notes
    -----
    The kernel regression is a non-parametric approach to estimate the
    conditional expectation of a random variable:

        E(Y|X) = f(X)

    where f is a non-parametric function. Based on the kernel density
    estimation, this code implements the Nadaraya-Watson kernel regression
    using the Gaussian kernel as follows:

        f(x) = sum(gaussian_kernel((x-X)/h).*Y)/sum(gaussian_kernel((x-X)/h))

    """
    ## Gaussian Kernel for continuous variables
    def __calc_gaussian_kernel(bw, X, x):
        return (np.sqrt(2*np.pi) ** -1) * np.exp(-.5 * ((X - x)/bw) ** 2)

    ## Calculate optimal bandwidth as suggested by Bowman and Azzalini (1997) p.31
    def __calc_optimal_bandwidth(X, Y):
        n = len(X)
        hx = np.median(abs(X - np.median(X)))/(0.6745*(4.0/3.0/n)**0.2)
        hy = np.median(abs(Y - np.median(Y)))/(0.6745*(4.0/3.0/n)**0.2)
        return np.sqrt(hy*hx)

    assert len(X) == len(Y), 'X and Y should have the same number of observations'

    optimal_bw = __calc_optimal_bandwidth(X, Y)
    assert optimal_bw > np.sqrt(np.spacing(1))*len(X), 'Based on the optimal bandwidth metric, there is no enough variation in the data. Regression is meaningless.'
    assert optimal_bw > bw, 'The specified bandwidth is higher than the optimal bandwidth'

    # remove nans from X
    Y = Y[np.logical_not(np.isnan(X))]
    X = X[np.logical_not(np.isnan(X))]
    # remove nans from Y
    X = X[np.logical_not(np.isnan(Y))]
    Y = Y[np.logical_not(np.isnan(Y))]

    n_obs = len(x)
    f_x = np.empty(n_obs)
    var_f_x = np.empty(n_obs)
    for i in xrange(n_obs):
        K = __calc_gaussian_kernel(bw, X, x[i])
        f_x[i] = (WEIGHT * Y * K).sum() / (WEIGHT * K).sum()
        var_f_x[i] = (WEIGHT * (Y ** 2) * K).sum() / (WEIGHT * K).sum()
    return f_x, var_f_x

def update_roll_adjusted_intraday_series(df, df_new, roll_date_str, before_roll_date_str, roll_adj):
    ''' function to roll adjut time series df and adding df_new to it
    Parameters
    ----------
    df: old dataframe, contains open high low close
    df_new: new dataframe, contains open high low close 
    before_roll_date_str: Date String. one business days before roll happens
    roll_date_str: Date String. Roll date. On the day, the contract will becomne new contract
    roll_adj: roll adjust amount. 0.0: means no adjustment
    ----------
    output:
    df_updated    
    
    '''
    df_adj = df.copy()
    df_adj.ix[:, ['Open', 'High', 'Low', 'Close']] =  df.ix[:, ['Open', 'High', 'Low', 'Close']].copy() + roll_adj
    df_updated = pd.concat([df_adj.ix[:before_roll_date_str], df_new.ix[roll_date_str:]], axis= 0)
    return df_updated

### Event functions
def get_event_intraday_data_outright(inputcsvdir, contracts, time_interval, time_int_n, time_zone, day_st, day_end, Events_dates, len_events):    
    event_vol_dict = {}
    event_data_dict = {}
    event_data_vol_adj_dict = {}
    df_variance_profile = {}
    #load intraday variance
    for i_contract in contracts:
        df_variance_profile[i_contract] =  pd.read_csv(inputcsvdir + '\\'+ i_contract+  '_intraday_variance_5min.csv', index_col=0)
        #load ATR 
        df_ATR = pd.read_csv(inputcsvdir + '\\'+ i_contract+ '_ATR.csv', index_col=0, parse_dates=True, infer_datetime_format= True)
        dates = list(df_ATR.index.date)
        #load 5 mins data for all trading days
        df_data_all = pd.read_csv(inputcsvdir + '\\'+ i_contract+ '_5min.csv', index_col= 0, parse_dates= True, infer_datetime_format= True) 
        df_data_all = df_data_all.tz_localize(time_zone)

        # load data matrix, event_data is a matrix whose rows are the evenet dates and col are intraday time interval 
        event_data = np.empty((len_events, (day_end-day_st)*len(time_interval)))
        event_data_date_time = np.tile(Events_dates[0], (len_events, (day_end-day_st)*len(time_interval)))
        event_vol = np.empty(len(Events_dates))
        for i,date_i in enumerate(Events_dates):
            # get location on dates, note dates are the index of df_ATR
            idx = dates.index(date_i)
            event_vol[i] = df_ATR.ewma[dates[idx + day_st -1]]  # day_start-1 as the vol
            for j, value in enumerate(range(day_st, day_end)):
               idx_tmp = idx + value
               df_tmp = get_data_by_date(df_data_all,dates,dates[idx_tmp])
               event_data[i, j*time_int_n:(j+1)*time_int_n]= df_tmp.values.T
               event_data_date_time[i, j*time_int_n:(j+1)*time_int_n] = list(df_tmp.index)

        # re-center end of T-1 day as zeros
        event_data = event_data[:, :] - event_data[:,(day_end-day_st)/2*time_int_n-1][:, np.newaxis]
        event_data_vol_adj = event_data /event_vol[:, np.newaxis]
        event_vol_dict[i_contract] = event_vol
        event_data_dict[i_contract] = event_data
        event_data_vol_adj_dict[i_contract] = event_data_vol_adj
    return event_vol_dict, df_variance_profile, event_data_dict, event_data_vol_adj_dict, event_data_date_time      

def get_event_spread_ratio_vol(inputcsvdir, spread, day_st, day_end, Event_date_index):
    spread_vol = np.empty(len(Event_date_index))
    spread_ratio = np.empty(len(Event_date_index))
    #load vol and ratio
    df_spread_ratio = pd.read_csv(inputcsvdir + '\\'+ spread + '_DV01_ratio.csv', index_col=0, parse_dates = True, infer_datetime_format= True)
    df_spread_vol = pd.read_csv(inputcsvdir + '\\'+ spread + '_Vol.csv', index_col=0, parse_dates = True, infer_datetime_format= True)

    for i,date_i in enumerate(Event_date_index): 
        loc_vol = df_spread_vol.index.get_loc(date_i)
        spread_vol[i] = df_spread_vol.ix[loc_vol + day_st -1, :]
        loc_ratio = df_spread_ratio.index.get_loc(date_i)
        spread_ratio[i] = df_spread_ratio.ix[loc_ratio + day_st -1, :]  
    return spread_vol, spread_ratio
    
def generate_pos_curve(mean, variance, ewma_seedperiod, window_size, smooth_window, section_st, section_ed):
    
    # ewma_seedperiod  --- row smooth window
    # window_size  ---- starting window for ewma
    # smooth_window --- column smooth window
    
    pos_raw = np.divide(mean, variance)
    pos_curve_1= np.zeros(pos_raw.shape)     # expanding windwow position
    pos_curve_2= np.zeros(pos_raw.shape)     # rolling window position
    x1 = np.arange(pos_raw.shape[1])    
    #rolling window kernal
    for i_row in np.arange(ewma_seedperiod, pos_curve_2.shape[0]):
        X = np.tile(x1, window_size)
        Y = pos_raw[i_row-window_size+1:i_row+1, :].flatten()
        f_x1, var_f_x1 = sig.calc_gaussian_kernel_regression(X,Y,x1, smooth_window)
        pos_curve_2[i_row, :] = f_x1/var_f_x1
        pos_curve_2[i_row, :] = pos_curve_2[i_row, :]/max(abs(pos_curve_2[i_row, section_st:section_ed]))     # normalize position curve
    #expanding window kernal
    for i_row in np.arange(0, pos_curve_1.shape[0]):
        X = np.tile(x1, i_row+1)
        Y = pos_raw[:i_row+1, :].flatten()
        f_x1, var_f_x1 = sig.calc_gaussian_kernel_regression(X,Y,x1, smooth_window)
        pos_curve_1[i_row, :] = f_x1/var_f_x1  
        pos_curve_1[i_row, :] = pos_curve_1[i_row, :]/max(abs(pos_curve_1[i_row, section_st:section_ed]))     # normalize position curve
    return pos_curve_1, pos_curve_2

def get_pl_pos(FS, gearing, cost_const, FPV, df_data, pos_curve, event_vol, Events_dates, data_date_time):
    df_pos_curve = pd.DataFrame(pos_curve, index = df_data.index)
    s_vol = pd.Series(data = event_vol, index= df_data.index)
    df_pos = df_pos_curve.shift(1).div(s_vol, axis = 0)*FS/gearing/FPV
    pl = df_pos.values*df_data.T.diff(1).T.values*FPV
    cost = cost_const*df_pos.T.diff(1).T.fillna(value = 0).abs().values*FPV
    pl_net = pl-cost
    #intraday PL
    df_pl_intraday = pd.DataFrame({'pl':pl.flatten(), 'pl_net': pl_net.flatten()},index = data_date_time.flatten())
    #Events PL
    df_pl_events = pd.DataFrame(data = pl, index = pd.DatetimeIndex(Events_dates))
    df_pl_events_net = pd.DataFrame(data = pl_net, index = pd.DatetimeIndex(Events_dates))
    # convert to daily PL
    daily_pl = df_pl_intraday.groupby(lambda x: x.date()).sum()
    daily_pl.index= pd.DatetimeIndex(daily_pl.index)
    calendar_dates = pd.date_range(start = daily_pl.index[0], end = daily_pl.index[-1], freq='B')
    daily_pl = daily_pl.reindex(index = calendar_dates, fill_value= 0)
           
    #intraday positon to end of day position
    df_pos_intraday = pd.DataFrame({'pos':df_pos.values.flatten()},index = data_date_time.flatten())
    daily_pos = df_pos_intraday.groupby(lambda x: x.date()).apply(lambda x: x.ix[-1,:])
    daily_pos.index= pd.DatetimeIndex(daily_pos.index)
    daily_pos = daily_pos.reindex(index = calendar_dates, fill_value= 0)
   
    return df_pl_intraday, daily_pl, df_pos_intraday, daily_pos, df_pl_events, df_pl_events_net 

######################################################################
##### plot functions
####################################################################### plot mean function
def plot_data_mean(data_dict, contracts, index_dates, column_times, plot_date_st, plot_date_ed, day_st, day_end, time_int_n):
    rcParams['figure.figsize'] = 18, 3
    ax1 = grid()
    hold(True)
    for i_contract in contracts:
        #convert to dataframe
        df_data = pd.DataFrame(data = data_dict[i_contract], index = index_dates, columns = column_times)
        df_data.ix[plot_date_st:plot_date_ed].mean(axis = 0).T.plot(ax = ax1, label = i_contract)
    for i in range(1,day_end - day_st):
        axvline(x = i* time_int_n-1, color = 'k')
    legend(loc = 'best') 
    return ax1
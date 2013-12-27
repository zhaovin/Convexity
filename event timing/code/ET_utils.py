# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:40:14 2013

@author: Y
"""
import pandas as pd
import numpy as np

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



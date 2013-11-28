# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:40:14 2013

@author: Y
"""
import pandas as pd

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
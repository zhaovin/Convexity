# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np

# <codecell>

df = pd.DataFrame(np.random.randn(200, 2), columns=['PRICE', 'VOLUME'], 
                  index=pd.date_range(start='2014-03-03-09:00:00', periods=200, freq='S'))
df['VOLUME'] = abs(df['VOLUME']) * 100

# <codecell>

print 'Head of synthetic tick data:'
print df.head()

# <codecell>

df_bar = df['PRICE'].resample('1Min', how='ohlc', closed='left')
df_bar = df_bar.join(df['VOLUME'].resample('1Min', how='sum', closed='left'))
print 'OHLC Minute Bar for first minute:'
print df_bar.ix['2014-03-03 09:00:00']

# <codecell>

first_minute = df.ix['2014-03-03 09:00:00':'2014-03-03 09:00:59'] # From :00 through :59  
first_minute_dict = {}
first_minute_dict['high'] = first_minute['PRICE'].max() # High is the maximum value
first_minute_dict['low'] = first_minute['PRICE'].min() # Low is the minimum value
first_minute_dict['open'] = first_minute['PRICE'].ix[0] # Open is the first value
first_minute_dict['close'] = first_minute['PRICE'].ix[-1] # Close is the last value
first_minute_dict['VOLUME'] = first_minute['VOLUME'].sum() # Sum of values

# <codecell>

manual_df = pd.Series(first_minute_dict)

# <codecell>

print 'OHLC Minute Bar for first minute:'
print manual_df

# <codecell>


# <codecell>



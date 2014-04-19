import os
import sys
import pandas as pd
import datetime
import math
import numpy as np
import scipy.optimize as op
import json
import ConfigParser




config = ConfigParser.RawConfigParser()
config.read(os.path.abspath('../../../idpresearch.cfg'))
sys.path.append(os.path.abspath(config.get('IDPResearch','path')))

import idpresearch.siglib as siglib
import idpresearch.risklib as risklib
import idpresearch.nbtestlib as nbtestlib
import idpresearch.datetimelib as datetimelib

class CA: 
    '''
    '''
    def __init__(self): 
        '''
        '''
        self.sig = siglib.Sig()
        self.risk = risklib.Risk()
        self.nbtest = nbtestlib.Backtest()
        self.dttime = datetimelib.Datetime()
        
    def buyAll(self, d_bond,d_equity,r_bond,r_equity):
        '''
        #this function solve the weight of spx in a buy all index to optimize a
        #backward looking sharpe ratio defined by total carry / vol in which
        #correlation is taken as last 100 days observation
        # d_equity: %ret of equity
        # d_bond: %ret of bond
        # r_equity - spx carry (dividend yield)
        # r_bond - ty carry (yield)
        # v_equity - spx variance
        # v_bond - ty variance
        # p - spx/ty correlation


        #result
        # x_IR - weight for maximizing IR
        # x_P - weight for minimizing variance assuming -50% constant correlation
        # x - average of the 2 above
        '''
        t=200
        t_s=30

        p=pd.rolling_corr(d_equity,d_bond,t)
        p=pd.ewma(p,halflife=t_s)

        p2 = pd.Series(-0.5, index=p.index)

        v_equity=pd.rolling_var(d_equity,t)
        v_bond=pd.rolling_var(d_bond,t)

        m=len(p)

        x_IR=p.copy()
        x_P=x_IR.copy()

        for i in range(0,m):

            f = lambda x, : -(x*r_equity[i]+(1-x)*r_bond[i])/np.sqrt((x**2*v_equity[i]+(1-x)**2*v_bond[i]+2*x*(1-x)*np.sqrt(v_equity[i]*v_bond[i])*p[i])*16)

            #fitting the data with fmin
            x0 = 0.2 # initial parameter value
            x1 = op.fminbound(f, 0.1,0.8,maxfun=100)
            x_IR[i]=x1
    
            #portfolio optimisation assuming a constant correlation of -50%
            f = lambda x, : -(x*r_equity[i]+(1-x)*r_bond[i])/np.sqrt((x**2*v_equity[i]+(1-x)**2*v_bond[i]+2*x*(1-x)*np.sqrt(v_equity[i]*v_bond[i])*p2[i])*16)

            # fitting the data with fmin
            x0 = 0.2 # initial parameter value
            x2 = op.fminbound(f, 0.1,0.8,maxfun=100)
            x_P[i]=x2
    
            w=(x_P+x_IR)/2
    
        return w


    
    def vol_ratio_index(self, vol_st, vol_lt, half_lifes, seed_period): 
        '''
        Calculates the vol term structure ratio and places a trade according to the applied rule.
        
        Parameters
        ----------
        vol_st : series
            Short term vol. 
        vol_lt : series 
            Long term vol. 
        half_lifes : list
            Half life periods for zscore calculation.
        seed_period : int
            Look-ahead period used to calculate the seed for zscore calculation.
        verbose : bool, optional
            Print diagnostics (defaults to False).
            
        Returns
        -------
        [signal, raw_signal] : list
            Signal and raw zscore series.
        
        '''
    
        raw = vol_st.div(vol_lt)
        
        raw_signal = self.zscore_mult(raw, half_lifes, seed_period)
        
        signal = raw_signal.apply(self.__ratio_rule)
        
        return [signal, raw_signal]
    
    def tanh_zscore_mult(self, ts,  half_lifes, seed_period): 
        '''
        Calculates the vol term structure ratio and places a trade according to the applied rule.
        
        Parameters
        ----------
        vol_st : series
            Short term vol. 
        vol_lt : series 
            Long term vol. 
        half_lifes : list
            Half life periods for zscore calculation.
        seed_period : int
            Look-ahead period used to calculate the seed for zscore calculation.
        verbose : bool, optional
            Print diagnostics (defaults to False).
            
        Returns
        -------
        [signal, raw_signal] : list
            Signal and raw zscore series.
        
        '''
    
        raw = ts
        
        raw_signal = self.zscore_mult(raw, half_lifes, seed_period)
        
        signal = raw_signal.apply(self.__ratio_rule)
        
        return signal
    
    
    def __ratio_rule(self, x): 
        '''
        If abs(zscore)<k0, index=0, else index=tanh((zscore-k0)/k1)*k2
        
        '''
        k0, k1, k2 = 1.2, 0.3, 2
        
        if abs(x)<k0: 
            return 0
        elif abs(x)>=k0: 
            return math.tanh(math.copysign(1, x)*((abs(x)-k0)/k1))*k2
    
    def zscore_mult(self, ser, half_lifes, seed_period):
        '''
        Calculates the average zscore for multiple half lifes. 
        
        Parameters
        ----------
        series : series
            Date-time indexed time series. 
        half_lifes : list
            Half life periods for zscore calculation.
        seed_period : int
            Look-ahead period used to calculate the seed for zscore calculation.
        verbose : bool, optional
            Print diagnostics (defaults to False).
            
        Returns
        -------
        zscore : series 
            Average zscore.
        
        '''
        
        
        df = pd.DataFrame(ser)
        signals = []
        
        for half_life in half_lifes: 
            data = self.sig.calc_zscore(df, 
                                        mean_halflife=half_life, 
                                        mean_seedperiod=seed_period
                                        )
            
            data.columns = [half_life]
            signals.append(data)
        
        signals = pd.concat(signals, axis=1)
        zscore = signals.mean(axis=1)
        
        return zscore
    
    
    def CA_TsInRange(self,rate, stdWindow,cap):
        '''
        # this function gives how far is the current rate compare to its trailing
        # range, first zscore of the current rate is calculated, then this is
        # mapped to a range score between [-2 2], with middle of the range (zcore ~ 0) mapped
        # to -2 and extreme range (zcore >2 or <-2) mapped to 2
        # cap =[N1,N2] caps the zscore
    
        # these two lines caps the zscore
        #zscoreRate(zscoreRate>capRange(2))=capRange(2);
        #zscoreRate(zscoreRate<capRange(1))=capRange(1);

        # linear mapping - y= (2/x0)*x-2
        # in which x0 is where neutral position is taken
        # assuming normal distribution, x0=0.66 --> 50% long 50% short
        # assuming normal distribution, x0=1 --> 33% long 66% short
        '''
        zscoreRate=self.sig.calc_zscore(rate,stdWindow,stdWindow)
        rscore_l = []
        x0_l=1
   
        rscore_l=(2/x0_l)*zscoreRate.abs()-2

        for idx, col in enumerate(rscore_l):
            if rscore_l[idx]<0:
                rscore_l[idx]=0
            if rscore_l[idx]>cap:
                rscore_l[idx]=cap


        x0_s=1
        rscore_s=(2/x0_s)*zscoreRate.abs()-2

        for idx, col in enumerate(rscore_s):
            if rscore_s[idx]>0:
                rscore_s[idx]=0
            if rscore_s[idx]<-cap:
                rscore_s[idx]=-cap

        rscore=rscore_l+rscore_s
        return rscore 
      
    def CA_TsInRange_MULT(self, rate):
        '''
         #window = np.array([10,50,50,50, 150])
          #weight = np.array([0,0, 0,0, 1])
          '''
        window=[10, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]


        weight=np.array([1,1,1,1,1,1,1,1,1,1,1,1],dtype=np.float64)/12

        rscore=np.zeros(len(rate))
    
        for i,iVal in enumerate(window):

            stdWindow=iVal
            zCap=2;

            signal = self.CA_TsInRange(rate,stdWindow,zCap)


            rscore=rscore+ weight[i]*signal

        return rscore 
       
    def CA_RiskInRange_MULT(self, rate):
        '''
        # this function returns a scaling indicator generated from where risk is in the range
        '''
        rscore = self.CA_TsInRange_MULT(rate)
        s=rscore.copy()
    
        for idx, val in enumerate(rscore):
            if rscore[idx]<0:
                s[idx]=1
            if rscore[idx]>=0:
                s[idx]=1-rscore[idx]
            if s[idx]<0:
                s[idx]=0
            
            

        return s

    
    def K_factor(self, sw, lw):
        '''
        #K factor calculation for osc function
        '''
        
        sw=float(sw)
        lw=float(lw)
        fs = 1- 1/sw
        fl = 1- 1/lw
        k = np.sqrt(fs**2/(1-fs**2)+ fl**2/(1-fl**2)- 2*fs*fl/(1-fs*fl))
    
        return k 
    
    def osc(self, x, sw, lw, vol):
        '''
        #   function to calculate the oscilator of a time series 
        #   y = osc(x, sw, lw, vol)
        #   x  --- input time series
        #   sw --- short window
        #   lw  --- long window
        #   vol  --- volatitility  used in the calculation
        '''
    
        ewma_s = pd.ewma(x, halflife=sw);
        ewma_l = pd.ewma(x, halflife=lw);

        ewma_diff = ewma_s - ewma_l;

        k = self.K_factor(sw, lw); 

        y = ewma_diff/(k*vol);
        return y
    
   

    def CA_tsMomFI(self, ts):
        '''
        #this function construct momentum signal for atimeseries
        #it returns the weighted signal and also individual oscilator reading
        #this is the mapping for the momentum indicator
        #map_h = @(x)x*exp(-x**2/4)*2.5;
        '''
        oscweight = np.array([25,  50, 25, 0, 0, 0],dtype=np.float64)/100
   
        rtn = ts - ts.shift(1);#bck looking ts ret
        ts=rtn.cumsum();

        #get the indicators, then map and then positions
    
        #for idx1, col1 in enumerate(swpnRets_back):
        rtn=pd.DataFrame(rtn)
        rtn.columns=['return']
        temp = self.risk.generate_ewma_riskmodel(rtn, 18, min_periods=10);   
        vol_18=temp.get('vol')
    
        price_osc=[]

        for i in range(1,7):
            sw = 2**(i-1)
            lw = 3*sw
            sw=float(sw)
            lw=float(lw)
            temp = self.osc(ts, sw, lw, vol_18)
            temp=pd.DataFrame(temp)
            temp.columns=[str(sw)+'_'+str(lw)]
            price_osc.append(temp)
    
        price_osc = pd.concat(price_osc, axis=1)
    

        p_r= price_osc*np.exp(-price_osc**2/4)*2.5

        weighted_p_r = p_r * (oscweight);
        mscore = weighted_p_r.sum(1); 
    
        return [ mscore , price_osc ]
 

    def CA_tsMomGen(self, ts, oscweight):

        rtn = ts - ts.shift(1);
        ts=rtn.cumsum();
 
        
        rtn=pd.DataFrame(rtn)
        rtn.columns=['return']
        temp = self.risk.generate_ewma_riskmodel(rtn, 18, min_periods=10)   
        vol_18=temp.get('vol')
    
        price_osc=[]

        for i in range(1,7):
            sw = 2**(i-1)
            lw = 3*sw
            sw=float(sw)
            lw=float(lw)
            temp = self.osc(ts, sw, lw, vol_18)
            temp=pd.DataFrame(temp)
            temp.columns=[str(sw)+'_'+str(lw)]
            price_osc.append(temp)
    
        price_osc = pd.concat(price_osc, axis=1)
    
        p_r= price_osc*np.exp(-price_osc**2/4)*2.5
    
       
        weighted_p_r = p_r * (oscweight)
        mscore = weighted_p_r.sum(1) 
    
        return [mscore , price_osc]
          
    def CA_stickyIVRV(self, rate, vol, rVolWindow,l_On,l_Off,s_On,s_Off):
        '''
         # this function calculate IVRV ratio and place trade according to the
         # following rules:
         #1) if ratio < l_On, entering long position
         #2) exit long position if ratio > l_off
         #3) if ratio > s_On, entering short position
         #4) exit short position if ratio < S_off
         # input vol here is blackvol
        '''
        rVol=pd.rolling_std(rate-rate.shift(1),rVolWindow);


        #so far we have annualised BP vol
        ann_rVol=rVol*(255**0.5);

        ann_bpVol=rate*vol;
        raw_Signal=ann_bpVol/ann_rVol;
    
        signal=raw_Signal.copy()

        for idx, val in enumerate(raw_Signal):
            if raw_Signal[idx]<l_On:
                signal[idx]=2
            elif raw_Signal[idx]>s_On:
                signal[idx]=-2
            else:
                signal[idx]=0
    
        for idx, val in enumerate(raw_Signal):
            if signal[idx-1]==2 and raw_Signal[idx]<l_Off:
                signal[idx]=2
            elif signal[idx-1]==-2 and raw_Signal[idx]>s_Off:
                signal[idx]=-2 
    

    
        return [signal, raw_Signal]
    
    
    def CA_volTermStr(self, vol_st, vol_lt,l_On,l_Off,s_On,s_Off):
        '''
        # this function calculate vol term structure ratio and place trade according to the
        # following rules:
        #1) if ratio < l_On, entering long position
        #2) exit long position if ratio > l_off
        #3) if ratio > s_On, entering short position
        #4) exit short position if ratio < S_off
        # input     vol here is bpvol
        # for now this function generate long only position by putting short
        # position as 0
        '''
        raw_Signal=vol_st/vol_lt

        signal=raw_Signal.copy()

        for idx, val in enumerate(raw_Signal):
            if raw_Signal[idx]<l_On:
                signal[idx]=2
            elif raw_Signal[idx]>s_On:
                signal[idx]=-2
            else:
                signal[idx]=0
    
        for idx, val in enumerate(raw_Signal):
            if signal[idx-1]==2 and raw_Signal[idx]<l_Off:
                signal[idx]=2
            elif signal[idx-1]==-2 and raw_Signal[idx]>s_Off:
                signal[idx]=-2 
    
        return [signal, raw_Signal]


    def CA_swaptionExMove(self, ret, sig1, sig2,sig3,thr):
        '''
        # this function generate the excessive signal
        # if the market move against model position on 2% tail, unwind position
        # put on 50% T+1
        # put on 100% T+2
        '''
        sigAggr=sig1+sig2+sig3;
        signal=ret.copy()
        m=len(ret);
        signal.values[:]=1

        for i in range(10,m):
    
            lvl1=(ret[1:i]).quantile(thr)
            lvl2=(ret[max(1,i-500):i]).quantile(1-thr)
            if ret[i]<lvl1:
                if sigAggr[i]>0:
                    signal[i]=0
                    if i<m-1:
                        signal[i+1]=0.5

            if ret[i]>lvl2:
                if sigAggr[i]<0:
                    signal[i]=0
                    if i<m-1:
                        signal[i+1]=0.5
        return signal
       

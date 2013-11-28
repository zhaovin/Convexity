    def calc_gaussian_kernel_regression(self, X, Y, x, bw = 0.05):
        """
        Calculate kernel regression using the gaussian kernel (regression type is local constant estimator).

        Parameters
        ----------
        X : list or array
            Independent variable
        Y : list or array
            Dependent variable
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
            f_x[i] = (Y * K).sum() / K.sum()
            var_f_x[i] = ((Y ** 2) * K).sum() / K.sum()
        return f_x, var_f_x
    
    def calc_ewm(self,
                  df,
                  halflife=252,
                  seedperiod=22,
                  scalevar=1):
        """
        Calculate exponentially-weighted-moving mean, variance, volatility, correlation and covariance

        Parameters
        ----------
        df : DataFrame
            DataFrame containing timeseries data (EWMA calculated on columns)
        halflife : int, optional
            Half-life period in days (defaults to 252)
        seedperiod : int, optional
            Look-ahead seed period in days (used to calculate the seed for EWMA) (defaults to 22)
        scalevar : int, optional
            Factor used to annualize variance (defaults to 1, assumes daily data and 252 points to calculate variance)

        Returns
        -------
        ewm_dict : dict
            Dictionary containing ewma mean, var, vol, cov and corr
        """
        self.log.debug('In function Dmat.calc_ewm')

        from idpresearch.risklib import Risk
        risk = Risk()
        ewm_dict = {}
        ewm_dict['mean'] = pd.ewma(df, com=risk.halflife_to_center_of_mass(halflife), min_periods=seedperiod)
        ewm_dict['var'] = pd.ewmvar(df, com=risk.halflife_to_center_of_mass(halflife), min_periods=seedperiod) * scalevar
        ewm_dict['vol'] = pd.ewmvol(df, com=risk.halflife_to_center_of_mass(halflife), min_periods=seedperiod) * np.sqrt(scalevar)
        ewm_dict['cov'] = risk.ewmcov_pairwise(df, halflife=halflife, min_periods=seedperiod)
        ewm_dict['corr'] = risk.ewmcorr_pairwise(df, halflife=halflife, min_periods=seedperiod)
        return ewm_dict    
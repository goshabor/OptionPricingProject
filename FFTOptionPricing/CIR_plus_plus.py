import numpy as np
import pandas as pd

from typing import List
from dataclasses import dataclass
from scipy.optimize import brentq, minimize
from scipy.stats import ncx2 #ncx2=non-central chi^2

### --- Svensson curve calibration ---
class calibrate_Svensson:
    def __init__(self, START_DATE, OIS_DATA, Svensson_init_params):
        self.OIS_DATA = OIS_DATA
        self.OIS_DATA['rate'] = OIS_DATA['rate_percent']/100

        self.START_DATE = START_DATE
        self.Svensson_init_params = Svensson_init_params

        df = self.compute_ZCB_curve(self.OIS_DATA, self.START_DATE)
        result = self.calibrate_Svensson_yield(df)

        self.df = df
        self.result = result
        self.Svensson_params = result.x
        
    ### --- Helper functions for computing the payoffs for OIS with different maturities --- 
    def create_timegrid_OIS(self, df, START_DATE):
        df = df.copy().sort_values('end').reset_index(drop=True)
        START_DATE = pd.to_datetime(START_DATE)

        df['delta_t'] = (df['end'] - START_DATE).dt.days / 365

        mask = df['tenor'].str.endswith('Y')
        increments = df.loc[mask, 'end'].diff().dt.days / 365

        df.loc[mask & (df['tenor']!='1Y'), 'delta_t'] = increments.loc[mask & (df['tenor']!='1Y')]

        return df
    
    def split_OIS_by_payoff(self, df):
        mask_single_payoff = df['tenor'].str.endswith(('W','M')) | (df['tenor'] == '1Y')
        df_single_payoff = df[mask_single_payoff].copy()

        df_special_payoff = df[df['tenor']=='18M'].copy()
        
        mask_multiple_payoffs = ~(mask_single_payoff | (df['tenor'] == '18M'))
        df_multiple_payoffs = df[mask_multiple_payoffs].copy()

        return df_single_payoff, df_special_payoff, df_multiple_payoffs
    
    def single_payoffs_ZCB(self, df_single_payoff):
        df = df_single_payoff.sort_values('end').reset_index(drop=True)
        df['ZCB'] = 1 / (1 + df['delta_t'] * df['rate'])
        return df
    
    def special_payoffs_ZCB(self, df_special_payoff, df_single_payoff):
        df = df_special_payoff.copy()
        one_year_tenor = df_single_payoff[df_single_payoff['tenor'] == '1Y'].iloc[0]

        df['ZCB'] = (1 - one_year_tenor['delta_t']*one_year_tenor['ZCB']*df['rate']) / (1 + (df['delta_t']-one_year_tenor['delta_t'])*df['rate'])
        return df
    
    def multiple_payoffs_ZCB(self, df_multiple_payoffs, df_single_payoff):
        df = df_multiple_payoffs.sort_values('end').reset_index(drop=True)
        
        one_year_tenor = df_single_payoff[df_single_payoff['tenor'] == '1Y'].iloc[0]
        temp = one_year_tenor['delta_t']*one_year_tenor['ZCB']

        ZCB = []

        for i in range(len(df)):
            P_n = (1 - df.loc[i,'rate']*temp) / (1 + df.loc[i,'rate']*df.loc[i,'delta_t'])
            ZCB.append(P_n)
            temp += df.loc[i,'delta_t']*P_n
        
        df['ZCB'] = ZCB
        return df
    
    def compute_ZCB_curve(self, df, START_DATE):
        df = self.create_timegrid_OIS(df, START_DATE)
        df['ZCB'] = np.nan

        df_single_payoff, df_special_payoff, df_multiple_payoffs = self.split_OIS_by_payoff(df)

        df1 = self.single_payoffs_ZCB(df_single_payoff)
        df2 = self.special_payoffs_ZCB(df_special_payoff, df1)
        df3 = self.multiple_payoffs_ZCB(df_multiple_payoffs, df1)
        
        df_temp = pd.concat([df1,df2], ignore_index=True)
        df = pd.concat([df_temp,df3], ignore_index=True).sort_values('end').reset_index(drop=True)

        df['time_to_maturity'] = (df['end'] - pd.to_datetime(START_DATE)).dt.days / 360
        df['yield'] = -np.log(df['ZCB'])/(df['time_to_maturity'])

        return df
    
    ### --- Instantaneous forward curve under Svensson framework ---
    def Svensson_forward_curve(self, t, Svensson_params):
        beta0, beta1, beta2, beta3, a1, a2 = Svensson_params
        return beta0 + beta1*np.exp(-a1*t) + beta2*a1*t*np.exp(-a1*t) + beta3*a2*t*np.exp(-a2*t)

    ### --- Yield curve under Svensson framework ---
    def Svensson_yield_curve(self, t, Svensson_params, eps=1e-12):
        beta0, beta1, beta2, beta3, a1, a2 = Svensson_params
        t_safe = np.where(t < eps, eps, t)

        return beta0 + beta1*(1-np.exp(-a1*t_safe))/(a1*t_safe) + beta2*((1-np.exp(-a1*t_safe))/(a1*t_safe) - np.exp(-a1*t_safe)) + beta3*((1-np.exp(-a2*t_safe))/(a2*t_safe) - np.exp(-a2*t_safe))
    
    ### --- Estimating parameters for Svensson ---
    def objective_func(self, Svensson_params, df):
        Svensson_yield = self.Svensson_yield_curve(t=df['time_to_maturity'], Svensson_params=Svensson_params)
        actual_yield = df['yield']
        return np.sum((Svensson_yield-actual_yield)**2)
    
    def calibrate_Svensson_yield(self, df):
        result = minimize(
            fun=self.objective_func,
            args=(df,),
            x0=self.Svensson_init_params,
            tol=1e-12
        )
        return result

### --- CIR ++ model calibration --- 
class CIR_plus_plus:
    def __init__(self, swaptions_df, OIS_df, START_DATE, VAL_DATE, Svensson_init_params, Theta_r, lambda_swaptions, IDX_TENOR, fixed_leg_dt):
        self.START_DATE = START_DATE
        self.VAL_DATE = VAL_DATE

        self.swaptions_df = swaptions_df
        self.lambda_swaptions = lambda_swaptions

        Svensson = calibrate_Svensson(START_DATE=self.START_DATE, OIS_DATA=OIS_df ,Svensson_init_params=Svensson_init_params)

        self.Svensson = Svensson
        self.Svensson_params = Svensson.Svensson_params
        self.Theta_r = Theta_r

        self.IDX_TENOR = IDX_TENOR
        self.fixed_leg_dt = fixed_leg_dt
    
    ### --- dataclasses to store swaptions --- 
    @dataclass
    class SwaptionInstrument:
        T_expiry: float
        T_arr: np.ndarray #coupon time after expiry
        X: float
        market_price: float
        market_vega: float
    
    ### --- Helper funtions for the creating Swaptions and Caps payoff schedules
    def tenor_to_years(self, string):
        string = string.strip().upper()
        if string.endswith('Y'): return float(string[:-1])
        if string.endswith('M'): return float(string[:-1]) / 12
        if string.endswith('W'): return float(string[:-1]) / 52
        if string.endswith('D'): return float(string[:-1]) / 365
        raise ValueError(f"Unrecognized tenor '{string}'")
    
    def make_schedule(self, start, end, step): 
        n = int(np.round((end - start) / step))
        grid = start + np.arange(n + 1) * step

        if(grid[-1] < end - 1e-12):
            grid = np.append(grid, end)
        else:
            grid[-1] = end
        return grid
    
    ### --- Deterministic shift --- 
    def CIR_forward_curve(self, t, Theta_r):
        x0, k, theta, sigma = Theta_r
        h = np.sqrt(k**2 + 2*sigma**2)
        
        term1 = (2*k*theta*(np.exp(t*h)-1)) / (2*h + (k+h)*(np.exp(t*h)-1))
        term2 = x0 * (4*h**2*np.exp(t*h)) / (2*h + (k+h)*(np.exp(t*h)-1))**2
        return term1 + term2

    def psi_CIR(self, t, Theta_r):
        f_M = self.Svensson.Svensson_forward_curve(t=t, Svensson_params=self.Svensson_params)
        f_CIR = self.CIR_forward_curve(t=t, Theta_r=Theta_r)
        return f_M - f_CIR
    
    ### --- Pricing of fixed-income instruments --- 
    def chi2_cdf(self, a, b, c):
        return ncx2.cdf(a, df=b, nc=c)
    
    def A(self, t, T, Theta_r):
        x0, kappa_r, theta_r, sigma_r = Theta_r
        h = np.sqrt(kappa_r**2 + 2*sigma_r**2)
        numerator = 2*h*np.exp(0.5*(kappa_r+h)*(T-t))
        denominator = 2*h + (kappa_r+h)*(np.exp((T-t)*h)-1)
        return (numerator/denominator)**(2*kappa_r*theta_r/sigma_r**2)
    
    def B(self, t, T, Theta_r):
        x0, kappa_r, theta_r, sigma_r = Theta_r
        h = np.sqrt(kappa_r**2 + 2*sigma_r**2)
        numerator = 2*(np.exp((T-t)*h)-1)
        denominator = 2*h + (kappa_r+h)*(np.exp((T-t)*h)-1)
        return numerator/denominator
    
    def A_hat(self, t, T, Theta_r):
        x0, kappa_r, theta_r, sigma_r = Theta_r
        Svensson_params = self.Svensson_params

        market_ZCB_price_t = np.exp(-self.Svensson.Svensson_yield_curve(t, Svensson_params) * t)
        market_ZCB_price_T = np.exp(-self.Svensson.Svensson_yield_curve(T, Svensson_params) * T)

        nominator = market_ZCB_price_T*self.A(0, t, Theta_r)*np.exp(-self.B(0, t, Theta_r)*x0)*self.A(t, T, Theta_r)*np.exp(self.B(t, T, Theta_r)*self.psi_CIR(t, Theta_r))
        denominator = market_ZCB_price_t*self.A(0, T, Theta_r)*np.exp(-self.B(0, T, Theta_r)*x0)
        return nominator/denominator
    
    # Zero-Coupon Bond
    def ZCB(self, t, T, Theta_r, r_t):
        return self.A_hat(t, T, Theta_r)*np.exp(-self.B(t, T, Theta_r)*r_t)
    
    #Put option on Zero-Coupon Bond
    def ZBP(self, t, T, tau, K, Theta_r, r_t):
        if T - t <= 1e-6:
            return max(K - self.ZCB(t, tau, Theta_r, r_t), 0.0)
        
        x0, kappa_r, theta_r, sigma_r = Theta_r
        
        h = np.sqrt(kappa_r**2 + 2*sigma_r**2)
        rho = 2*h / (sigma_r**2 * (np.exp(h*(T-t))-1))
        psi = (kappa_r+h)/sigma_r**2

        market_ZCB_price_T = np.exp(-self.Svensson.Svensson_yield_curve(T, self.Svensson_params) * T)
        market_ZCB_price_tau = np.exp(-self.Svensson.Svensson_yield_curve(tau, self.Svensson_params) * tau)

        r_hat = (np.log(self.A(T,tau,Theta_r)/K) - np.log((market_ZCB_price_T*self.A(0, tau, Theta_r) * np.exp(-self.B(0, tau, Theta_r)*x0))/(market_ZCB_price_tau*self.A(0, T, Theta_r)*np.exp(-self.B(0, T, Theta_r)*x0)))) / self.B(T, tau, Theta_r)

        a1 = 2*r_hat*(rho+psi+self.B(T, tau, Theta_r))
        a2 = 2*r_hat*(rho+psi)
        b = (4*kappa_r*theta_r)/sigma_r**2
        c1 = 2*rho**2 * (r_t - self.psi_CIR(t, Theta_r)) * np.exp(h*(T-t)) / (rho + psi+self.B(T, tau, Theta_r))
        c2 = 2*rho**2 * (r_t - self.psi_CIR(t, Theta_r)) * np.exp(h*(T-t)) / (rho + psi)

        term1 = K * self.ZCB(t, T, Theta_r, r_t) * (1-self.chi2_cdf(a2, b, c2))
        term2 = self.ZCB(t, tau, Theta_r, r_t) * (1-self.chi2_cdf(a1, b, c1))
        return term1 - term2
    
    #swaption helpers and pricer
    def swap_PV_at_expiry(self, T, T_arr, X, Theta_r, r_star):
        total = 0
        tau_arr = np.diff(T_arr)

        for i in range(len(tau_arr)):
            X_i = self.A_hat(T, T_arr[i+1], Theta_r) * np.exp(-self.B(T, T_arr[i+1], Theta_r)*r_star)
            total += X * tau_arr[i] * X_i
        
        X_n = self.A_hat(T, T_arr[-1], Theta_r) * np.exp(-self.B(T, T_arr[-1], Theta_r) * r_star) 
        total += X_n
        return total
    
    def find_r_star(self, T, T_arr, X, Theta_r):
        def F(r): return self.swap_PV_at_expiry(T, T_arr, X, Theta_r, r) - 1

        a, b = -1, 1
        for i in range(1000):
            if(F(a) * F(b) < 0): break
            a -= 0.01
            b += 0.01
            
        return brentq(F, a, b, maxiter=10000)
    
    def payer_swaption_price(self, t, T, T_arr, N, X, Theta_r, r_t):
        r_star = self.find_r_star(T, T_arr, X, Theta_r) #NOTE: r_star comes from Jamshidan decomposition
        tau_arr = np.diff(T_arr)
        swaption_price = 0

        for i in range(len(tau_arr)):
            X_i = self.A_hat(T, T_arr[i+1], Theta_r) * np.exp(-self.B(T, T_arr[i+1], Theta_r)*r_star)
            swaption_price += X * tau_arr[i] * self.ZBP(t, T, T_arr[i+1], X_i, Theta_r, r_t) #coupon: c_i = X*tau_i
        
        #last coupon payment: c_n = 1+X*tau_n (X*tau_n is already taken into account by for-loop, but the +1 part remains)
        X_n = self.A_hat(T, T_arr[-1], Theta_r) * np.exp(-self.B(T, T_arr[-1], Theta_r) * r_star)
        swaption_price += self.ZBP(t, T, T_arr[-1], X_n, Theta_r, r_t)
        
        return N*swaption_price
    
    #short rate at time 0
    def short_rate0(self, Theta_r, Svensson_params):  # r(0) = x(0) + phi(0)
        eps = 1e-8
        f_mkt = self.Svensson.Svensson_forward_curve(eps, Svensson_params)
        f_cir = self.CIR_forward_curve(np.array([eps]), Theta_r)[0]
        return Theta_r[0] + f_mkt - f_cir
    
    ### --- Building caps and swaption payoffs (dataclass) from pandas dataframe ---
    def build_swaptions_from_df(self)-> List[SwaptionInstrument]: #fixed_leg_dt=1 --> annual fixed leg
        VAL_DATE = pd.to_datetime(self.VAL_DATE)
        instrument = []

        for i, row in self.swaptions_df.iterrows():
            T = (pd.to_datetime(row['expiry']) - VAL_DATE).days / 365
            tenor_years = self.tenor_to_years(str(row['tenor']))
            T_arr = self.make_schedule(T, T + tenor_years, self.fixed_leg_dt)
            X = float(row['strike_pct']) / 100
            
            price = float(row['premium_pct']) / 100 
            vega = float(row['vega_pct']) / 100
            instrument.append(self.SwaptionInstrument(T_expiry=T, T_arr=T_arr, X=X, market_price=price, market_vega=vega))

        return instrument
    
    ### --- residual function for minimization ---
    def objective_func(self, Theta_r, swaptions_dataclass):
        r0 = self.short_rate0(Theta_r=Theta_r, Svensson_params=self.Svensson_params)
        
        resids_swaptions = []
        for swaption in swaptions_dataclass:
            model_swaption_price = self.payer_swaption_price(0, swaption.T_expiry, swaption.T_arr, 1, swaption.X, Theta_r, r0)  #1 is notional amount
            weighted_residual_swaption = self.lambda_swaptions * (model_swaption_price - swaption.market_price)**2
            resids_swaptions.append(weighted_residual_swaption)
        return np.mean(np.array(resids_swaptions))

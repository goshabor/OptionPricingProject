import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.lib.scimath import sqrt as csqrt

### --- Monte Carlo Pricing Method ---
class OptionPricingMonteCarlo:
    def __init__(self, t, T, P_t_T, init_params, Svensson_params, short_rate_params, vola_params, MC_params):
        self.t = t #valuation date
        self.T = T #time of maturity
        self.tau = T-t #time to maturity
        self.P_t_T = P_t_T #P(t,T)

        self.S_init, self.r_init, self.nu_init, self.K_grid = init_params

        self.beta0, self.beta1, self.beta2, self.beta3, self.a1, self.a2 = Svensson_params
        self.x0, self.kappa_r, self.theta_r, self.sigma_r = short_rate_params
        self.h = np.sqrt(self.kappa_r**2 + 2*self.sigma_r**2)

        self.kappa_nu, self.theta_nu, self.sigma_nu, self.rho = vola_params

        self.N_time_grid, self.N_paths = MC_params
        self.dt = (T-t)/MC_params[0]

    def hockey_stick_func(self, x):
        return np.maximum(x,0)
    
    def generate_Brownian_motion(self):
        rng = np.random.default_rng()
        corr_matrix = np.array([[1, 0, self.rho],
                                [0, 1, 0],
                                [self.rho, 0, 1]])
        L = np.linalg.cholesky(corr_matrix)

        Z = rng.standard_normal(size=(3,self.N_paths,self.N_time_grid))
        X = np.einsum('ij,jpt->ipt', L, Z)
        return np.sqrt(self.dt)*X
    
    def psi_CIR(self, t):
        f_Svensson = self.beta0 + self.beta1*np.exp(-self.a1*t) + self.beta2*self.a1*t*np.exp(-self.a1*t) + self.beta3*self.a2*t*np.exp(-self.a2*t)
        
        exp_term = np.exp(t*self.h) - 1
        sum_term = self.kappa_r + self.h
        temp1 = 2*self.kappa_r*self.theta_r*exp_term / (2*self.h + sum_term*exp_term)
        temp2 = self.x0 * 4*self.h**2*np.exp(t*self.h) / (2*self.h + sum_term*exp_term)**2
        f_CIR = temp1 + temp2

        return f_Svensson - f_CIR
    
    def psi_CIR_deriv(self, t):
        f_Svensson_deriv = -self.a1*self.beta1*np.exp(-self.a1*t) + self.beta2*self.a1*np.exp(-self.a1*t)*(1-self.a1*t) + self.beta3*self.a2*np.exp(-self.a2*t)*(1-self.a2*t)
        
        common_nominator = 4*self.h**2*np.exp(t*self.h)
        common_denominator = (self.kappa_r + self.h)*np.exp(t*self.h) - self.kappa_r + self.h
        f_CIR_deriv = self.kappa_r*self.theta_r*common_nominator/common_denominator**2 + self.x0 * common_nominator*self.h*((self.kappa_r+self.h)*np.exp(t*self.h) + self.kappa_r - self.h)/common_denominator**3

        return f_Svensson_deriv - f_CIR_deriv
    
    def B(self, t):
        exp_term = np.exp((self.T-t)*self.h) - 1
        return (2*exp_term) / (2*self.h + (self.kappa_r+self.h)*exp_term)
    
    def Monte_Carlo_Call_Price(self):
        dt = self.dt
        dW = self.generate_Brownian_motion()
        dW_S, dW_r, dW_nu = dW

        call_prices = []
        time_grid = np.linspace(self.t, self.T, self.N_time_grid+1)

        S = np.zeros((self.N_paths, self.N_time_grid+1))
        S[:,0] = self.S_init

        r = np.zeros((self.N_paths, self.N_time_grid+1))
        r[:,0] = self.r_init

        nu = np.zeros((self.N_paths, self.N_time_grid+1))
        nu[:,0] = self.nu_init

        eps = 1e-12

        for n in range(int(self.N_time_grid)):
            t_n = time_grid[n] 
            S[:,n+1] = S[:,n] * (1 + r[:,n]*dt + np.sqrt(np.maximum(nu[:,n], eps))*dW_S[:,n])
            r[:,n+1] = r[:,n] + (self.kappa_r*(self.theta_r - r[:,n] + self.psi_CIR(t_n)) + self.psi_CIR_deriv(t_n) - self.sigma_r**2*self.B(t_n)*(r[:,n]-self.psi_CIR(t_n)))*dt + self.sigma_r*np.sqrt(np.maximum(r[:,n]-self.psi_CIR(t_n), eps))*dW_r[:,n]
            nu[:,n+1] = nu[:,n] + self.kappa_nu*(self.theta_nu-nu[:,n])*dt + self.sigma_nu*np.sqrt(np.maximum(nu[:,n], eps))*dW_nu[:,n]
        
        for K in self.K_grid:
            call_price_k = self.P_t_T * np.mean(self.hockey_stick_func(S[:,-1]-K))
            call_prices.append(call_price_k)
        
        return np.array(call_prices)

### --- FFT Pricing Method as in Carr and Madan (1999) ---
class OptionPricingCarrMadan:
    def __init__(self, t, T, P_t_T, init_params, Svensson_params, Theta_r, Theta_nu, CarrMadan_params, n_tau):
        #initialising inputs
        self.t = t
        self.T = T
        self.tau = T-t
        self.P_t_T = P_t_T

        #initial parameters for stock, short rate and strike price
        self.S_init, self.r_init, self.K = init_params

        #short rate parameters
        self.beta0, self.beta1, self.beta2, self.beta3, self.a1, self.a2 = Svensson_params
        self.x0, self.kappa_r, self.theta_r, self.sigma_r = Theta_r
        self.h = np.sqrt(self.kappa_r**2 + 2.0*self.sigma_r**2)
        
        #volatility parameters
        self.kappa_nu, self.theta_nu, self.sigma_nu, self.nu_init, self.rho = Theta_nu 

        #Carr and Madan FFT parameters
        self.eta, self.N_FFT, self.alpha = CarrMadan_params

        ### --- Putting helper for the FFT into the cache ---
        self.u_grid = self.eta*np.arange(self.N_FFT)
        self.shifted_grid = self.u_grid - 1j*(1+self.alpha)

        self.lambda_ = 2*np.pi/(self.N_FFT*self.eta)
        self.epsilon = np.pi/self.eta

        self.k_grid = -self.epsilon + self.lambda_*np.arange(self.N_FFT)
        self.K_grid = np.exp(self.k_grid) * self.S_init

        #Simpson's rule weights
        W = np.ones(self.N_FFT)
        W[1:self.N_FFT-1:2] = 4
        W[2:self.N_FFT-2:2] = 2
        self.W_Simpson = W*self.eta/3
        
        #miscellaneous for the FFT
        self.phase = np.exp(1j*self.epsilon*self.u_grid)
        self.call_weight = np.exp(-self.alpha*self.k_grid)/np.pi
        self.denominator = self.alpha**2 + self.alpha - self.u_grid**2 + 1j*self.u_grid*(2*self.alpha+1)

        #tau-grid
        self.n_tau = n_tau
        self.build_tau_cache(n_tau=self.n_tau)
    
    ### --- Helper functions ---
    def stable_sqrt(self, z):
        d = csqrt(z)
        return np.where(np.real(d)>=0, d, -d)

    def psi_CIR(self, t):
        f_Svensson = self.beta0 + self.beta1*np.exp(-self.a1*t) + self.beta2*self.a1*t*np.exp(-self.a1*t) + self.beta3*self.a2*t*np.exp(-self.a2*t)
        
        exp_term = np.exp(t*self.h) - 1
        sum_term = self.kappa_r + self.h
        temp1 = 2*self.kappa_r*self.theta_r*exp_term / (2*self.h + sum_term*exp_term)
        temp2 = self.x0 * 4*self.h**2*np.exp(t*self.h) / (2*self.h + sum_term*exp_term)**2
        f_CIR = temp1 + temp2

        return f_Svensson - f_CIR
    
    def psi_CIR_deriv(self, t):
        f_Svensson_deriv = -self.a1*self.beta1*np.exp(-self.a1*t) + self.beta2*self.a1*np.exp(-self.a1*t)*(1-self.a1*t) + self.beta3*self.a2*np.exp(-self.a2*t)*(1-self.a2*t)
        
        common_nominator = 4*self.h**2*np.exp(t*self.h)
        common_denominator = (self.kappa_r + self.h)*np.exp(t*self.h) - self.kappa_r + self.h
        f_CIR_deriv = self.kappa_r*self.theta_r*common_nominator/common_denominator**2 + self.x0 * common_nominator*self.h*((self.kappa_r+self.h)*np.exp(t*self.h) + self.kappa_r - self.h)/common_denominator**3

        return f_Svensson_deriv - f_CIR_deriv
    
    def B(self, tau):
        exp_term = np.exp(tau*self.h) - 1.0
        return (2*exp_term) / (2*self.h + (self.kappa_r+self.h)*exp_term)
    
    ### --- Vectorising calls used for the Riccati ODEs ---
    def build_tau_cache(self, n_tau):
        self.tau_grid = np.linspace(0, self.tau, n_tau)
        self.dt = self.tau_grid[1] - self.tau_grid[0]
        self.t_grid = self.T - self.tau_grid

        B_tau = self.B(self.tau_grid)
        psi_CIR = self.psi_CIR(self.t_grid)
        psi_CIR_deriv = self.psi_CIR_deriv(self.t_grid)

        #parameters for Psi_1_1
        self.a_Psi_1_1 = self.sigma_r**2/2
        self.b_Psi_1_1 = - (self.kappa_r + self.sigma_r**2*B_tau)

        #parameters for Phi_1
        self.a_Phi_1 = -0.5*self.sigma_r**2 * psi_CIR
        self.b_Phi_1 = self.kappa_r*(self.theta_r+psi_CIR) + psi_CIR_deriv + self.sigma_r**2*psi_CIR*B_tau
        self.c_Phi_1 = self.kappa_nu*self.theta_nu
        
        #Simpson's weights on tau for accurate tau-integrals
        W = np.ones(n_tau)
        W[1:n_tau-1:2] = 4
        W[2:n_tau-2:2] = 2
        self.W_Simpson_tau = W*self.dt/3

    def RK4_Psi_1_1(self, u_grid):
        N = u_grid.shape[0]
        y = np.zeros(N, dtype=np.complex128)
        
        y_path = np.empty((self.n_tau, N), dtype=np.complex128)
        y_path[0, :] = y

        dt = self.dt

        for k in range(self.n_tau-1):
            a_Psi_1_1 = self.a_Psi_1_1
            b_Psi_1_1 = self.b_Psi_1_1[k]
            c_Psi_1_1 = 1j*u_grid

            def Psi_1_1(z):
                return a_Psi_1_1*z*z + b_Psi_1_1*z + c_Psi_1_1
            
            #Runge-Kutta 4 method
            k1 = Psi_1_1(y)
            k2 = Psi_1_1(y + 0.5*dt*k1)
            k3 = Psi_1_1(y + 0.5*dt*k2)
            k4 = Psi_1_1(y + dt*k3)
            y = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

            y_path[k+1,: ] = y
        
        return y_path

    def Psi_1_2(self, tau, u):
        beta_nu = self.kappa_nu - 1j*u*self.rho*self.sigma_nu
        d_nu = self.stable_sqrt(beta_nu**2 + self.sigma_nu**2*(u**2+1j*u))
        g_nu = (beta_nu - d_nu) / (beta_nu + d_nu)

        exp_term = np.exp(-tau*d_nu)
        term1 = (beta_nu - d_nu) / self.sigma_nu**2
        term2 = (1-exp_term)/ (1-g_nu*exp_term)
        return term1*term2
    
    def characteristic_func_helper(self, u_grid):
        y_path = self.RK4_Psi_1_1(u_grid=u_grid)
        Psi_1_1_u = y_path[-1,:]

        Psi_1_2_grid = self.Psi_1_2(self.tau_grid[:, None], u_grid[None, :])
        Psi_1_2_u = Psi_1_2_grid[-1, :]

        integrand_base = self.a_Phi_1[:, None]*(y_path**2) + self.b_Phi_1[:,None]*y_path
        Phi_1_u = (self.W_Simpson_tau[:,None]*integrand_base).sum(axis=0) + self.c_Phi_1*(self.W_Simpson_tau[:,None]*Psi_1_2_grid).sum(axis=0)

        phi_Y = np.exp(Phi_1_u + Psi_1_1_u*self.r_init + Psi_1_2_u*self.nu_init)
        return phi_Y
    
    def characteristic_func_1_Y(self, u):
        u_grid = np.atleast_1d(u).astype(np.complex128)
        phi_arr = self.characteristic_func_helper(u_grid)
        return phi_arr[0] if np.ndim(u)==0 else phi_arr
    
    def Carr_Madan_FFT_Call_Prices(self):
        phi_arr = self.characteristic_func_helper(self.shifted_grid)
        c_tilde = self.S_init*self.P_t_T*phi_arr/self.denominator

        FFT_input = self.phase*c_tilde*self.W_Simpson
        C_grid = self.call_weight*np.real(np.fft.fft(FFT_input))
        return self.K_grid, C_grid
    
### --- Calibrate FFT Option pricing ---
class CalibrationFFT:
    def __init__(self, df, S_t, t, r_t, VAL_DATE, Theta_r, Svensson_params, Theta_nu_init, CarrMadan_params, n_tau):
        ### --- Initialising variables ---
        self.t = t
        self.S_t = S_t
        self.r_t = r_t
        self.VAL_DATE = VAL_DATE

        self.Theta_r = Theta_r
        self.Svensson_params = Svensson_params
        self.Theta_nu_init = Theta_nu_init

        self.CarrMadan_params = CarrMadan_params
        self.n_tau = n_tau

        ### --- Data pre-processing ---
        df = df.copy()

        df['time_to_maturity_years'] = pd.to_numeric(df['time_to_maturity_years'], errors='coerce')
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')

        maturities = np.sort(df['time_to_maturity_years'].unique())
        strike_grid = np.sort(df['strike'].unique()) #contains all unique strikes, needed to inialize a 3D-array containing all maturities, strikes and call prices available in the market data

        relevant_columns = ['close']
        data_numpy = np.full((len(maturities), len(strike_grid), len(relevant_columns)+1), np.nan, dtype=float)
        data_numpy[:,:,0] = strike_grid[None,:]

        for i, T in enumerate(maturities):
            mask = np.isclose(df['time_to_maturity_years'].values, T, atol=1e-8, rtol=0)
            sub_data = df.loc[mask, ['strike']+relevant_columns]

            if(sub_data.empty): continue

            strikes = sub_data['strike'].to_numpy()

            column_idx = np.searchsorted(strike_grid, strikes)
            mask2 = (column_idx < len(strike_grid)) & np.isclose(strike_grid[column_idx], strikes, atol=1e-8, rtol=0)
            
            strikes = strikes[mask2]
            column_idx = column_idx[mask2]
            sub_data = sub_data.iloc[mask2]

            for j, col in enumerate(relevant_columns, start=1):
                data_numpy[i, column_idx, j] = sub_data[col].to_numpy()
            
        self.maturities = maturities
        self.data_numpy = data_numpy
        self.strikes = strike_grid

        ### --- Pre-computing discount factors using the CIR++ model (see Brigo, Mercurio, Ch.3.9) ---
        self.P_t_T = np.array([self.P(self.t, T) for T in maturities])
        
        ### --- Building one FFT pricer per maturity ---
        self.per_T = [] #list of dicts: 'T':..., 'pricer':..., 'idx':..., 'w':..., 'price':...}
        for i, T in enumerate(maturities):
            pricer = OptionPricingCarrMadan(
                t=t, 
                T=T, 
                P_t_T=self.P_t_T[i],
                init_params=[self.S_t, self.r_t, self.S_t],
                Svensson_params=self.Svensson_params,
                Theta_r=self.Theta_r,
                Theta_nu=self.Theta_nu_init, #updated per eval in the minimizer
                CarrMadan_params=self.CarrMadan_params,
                n_tau=self.n_tau
            )

            #selecting market data available for this maturity
            data = self.data_numpy[i]

            cols_needed = [0, 1]  #strike, price
            rows_ok = ~np.isnan(data[:, cols_needed]).any(axis=1)
            data_clean = data[rows_ok]
            if data_clean.size == 0:
                continue

            strike_i = data_clean[:, 0]
            close_i  = data_clean[:, 1]    

            #interpolation mapping strike -> FFT grid
            K_grid = pricer.K_grid
            kmin, kmax = K_grid[0], K_grid[-1]
            in_range = (strike_i >= kmin) & (strike_i <= kmax)
            if not np.any(in_range):
                continue

            k_use = strike_i[in_range]
            left_idx = np.searchsorted(K_grid, k_use) - 1
            left_idx = np.clip(left_idx, 0, len(K_grid) - 2)

            K_left  = K_grid[left_idx]
            K_right = K_grid[left_idx + 1]
            w = (k_use - K_left) / (K_right - K_left)

            self.per_T.append({
                'T': float(T),
                'pricer': pricer,
                'idx': left_idx.astype(int),
                'w': w.astype(float),
                'close': close_i[in_range].astype(float)
            })
    
    ### --- Fixed-income ---
    def P(self, t, T):
        beta0, beta1, beta2, beta3, a1, a2 = self.Svensson_params
        x0, kappa_r, theta_r, sigma_r = self.Theta_r

        h = np.sqrt(kappa_r**2 + 2*sigma_r**2)

        def Svensson_forward_curve(tt):
            return beta0 + beta1*np.exp(-a1*tt) + beta2*a1*tt*np.exp(-a1*tt) + beta3*a2*tt*np.exp(-a2*tt)

        def Svensson_yield_curve(tt, eps=1e-12):
            t_safe = np.where(tt < eps, eps, tt)
            return beta0 + beta1*(1-np.exp(-a1*t_safe))/(a1*t_safe) + beta2*((1-np.exp(-a1*t_safe))/(a1*t_safe) - np.exp(-a1*t_safe)) + beta3*((1-np.exp(-a2*t_safe))/(a2*t_safe) - np.exp(-a2*t_safe))
        
        def P_market(tt):
            return np.exp(-Svensson_yield_curve(tt)*np.array(tt, ndmin=1))
        
        def CIR_forward_curve(tt):
            term1 = (2*kappa_r*theta_r*(np.exp(tt*h)-1)) / (2*h + (kappa_r+h)*(np.exp(tt*h)-1))
            term2 = x0 * (4*h**2*np.exp(tt*h)) / (2*h + (kappa_r+h)*(np.exp(tt*h)-1))**2
            return term1 + term2
        
        def psi_CIR(tt):
            tt = np.array(tt, ndmin=1)
            f_M = Svensson_forward_curve(tt)
            f_CIR = CIR_forward_curve(tt)
            return f_M - f_CIR
        
        def A(tt, TT):
            numerator = 2*h*np.exp(0.5*(kappa_r+h)*(TT-tt))
            denominator = 2*h + (kappa_r+h)*(np.exp((TT-tt)*h)-1)
            return (numerator/denominator)**(2*kappa_r*theta_r/sigma_r**2)
            
        def B(tt, TT):
            numerator = 2*(np.exp((TT-tt)*h)-1)
            denominator = 2*h + (kappa_r+h)*(np.exp((TT-tt)*h)-1)
            return numerator/denominator

        def A_hat(tt, TT):
            market_ZCB_price_t = P_market(tt)
            market_ZCB_price_T = P_market(TT)

            nominator = market_ZCB_price_T*A(0,tt)*np.exp(-B(0,tt)*x0)*A(tt,TT)*np.exp(B(tt,TT)*psi_CIR(tt))
            denominator = market_ZCB_price_t*A(0,TT)*np.exp(-B(0,TT)*x0)
            return nominator/denominator
        
        return A_hat(t,T)*np.exp(-B(t,T)*self.r_t)
    
    ### --- Function to be minimized ---
    def objective_func(self, Theta_nu, lambda_nu, penalty=False):
        BIG = 1e30
        kappa_nu, theta_nu, sigma_nu, nu_t, rho = Theta_nu

        total_error = 0
        for entry in self.per_T:
            pricer = entry['pricer']
            
            pricer.kappa_nu = kappa_nu
            pricer.theta_nu = theta_nu
            pricer.sigma_nu = sigma_nu
            pricer.nu_init = nu_t
            pricer.rho = rho
            pricer.c_Phi_1 = pricer.kappa_nu*pricer.theta_nu

            _, C_grid = pricer.Carr_Madan_FFT_Call_Prices()

            #interpolating to market strikes using pre-computed idx and weight
            idx = entry['idx']
            weight = entry['w']
            
            C_left = C_grid[idx]
            C_right = C_grid[idx+1]
            C_interpolated = (1-weight)*C_left + weight*C_right

            close = entry['close']

            rel_error = (C_interpolated-close)/close
            total_error += np.mean(rel_error**2)

            if(not np.isfinite(total_error) or np.isnan(total_error)):
                return BIG
        
        return lambda_nu*total_error

### ------ splits data into test and training sets ------
def split_test_training_df(filepath, cutoff_year_lower, cutoff_year_upper, moneyness_pct_lower=75, moneyness_pct_upper=100, test_size=0.1, random_state=42):
    data = pd.read_excel(filepath)
    data = data.rename(columns={'STRIKE_PRC': 'strike', 'VEGA': 'vega', 'CF_CLOSE': 'close'})
    
    data = data.dropna(subset=['vega'])
    data = data.dropna(subset=['close'])

    S0 = float(data['S0'].unique()[0])
    strike_cutoff_lower = S0*(1-moneyness_pct_lower/100)
    strike_cutoff_upper = S0*(1+moneyness_pct_upper/100)

    mask_year = ((data['time_to_maturity_years'] >= cutoff_year_lower) & (data['time_to_maturity_years'] <= cutoff_year_upper))
    mask_strike = ((data['strike'] >= strike_cutoff_lower) & (data['strike'] <= strike_cutoff_upper))

    df = data[mask_year & mask_strike].copy()

    test_df = df.groupby(df['time_to_maturity_years'], group_keys=False).sample(frac=test_size, random_state=random_state)
    train_df = df.drop(test_df.index)
    
    return train_df, test_df, S0

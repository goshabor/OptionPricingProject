import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch import optim

USE_MPS = torch.backends.mps.is_available()
device = torch.device('mps' if USE_MPS else ('cuda' if torch.cuda.is_available() else 'cpu'))
torch.set_default_dtype(torch.float32 if device.type=='mps' else torch.float64)

# device = torch.device('cpu')

### ------ PINN ------
class PINN(nn.Module):
    def __init__(self, strike_price, Theta_r, Svensson_params, bounds_S, bounds_r, bounds_nu, bounds_rho, bounds_time, PINN_params, boundary_loss_weights):
        super().__init__()
        # bounds_S = [S_min, S_max]
        # Theta_r = [x_0, kappa_r, theta_r, sigma_r]
        # EM_params = [n_time_grid]
        # bounds_nu = [[nu_min, nu_max], [kappa_nu_min, kappa_nu_max], [theta_nu_min, theta_nu_max], [sigma_nu_min, sigma_nu_max]]
        # bounds_rho = [rho_min, rho_max]
        # bounds_time = [T_min, T_max]
        # Svensson_params = [beta0, beta1, beta2, beta3, a1, a2]
        # PINN_params = [input_dim, hidden_dim, num_layers, batch_boundary, batch_interior]
        # boundary_loss_weights = [weights_interior, weights_lower, weights_upper, weights_terminal]

        #bounds for stock prices
        self.S_min, self.S_max = bounds_S
        self.K = strike_price

        #short rates params
        self.r_min, self.r_max = bounds_r
        self.Theta_r = Theta_r 
        self.h = float(math.sqrt(self.Theta_r[1]**2 + 2*Theta_r[-1]**2))

        #parameters for Svensson curve
        self.Svensson_params = Svensson_params
        self.beta0, self.beta1, self.beta2, self.beta3, self.a1, self.a2 = Svensson_params
        
        #bounds for volatility model
        self.nu_min, self.nu_max = bounds_nu[0]
        self.kappa_nu_min, self.kappa_nu_max = bounds_nu[1]
        self.theta_nu_min, self.theta_nu_max = bounds_nu[2]
        self.sigma_nu_min, self.sigma_nu_max = bounds_nu[3]

        #bounds for correlation and time to maturity
        self.rho_min, self.rho_max = bounds_rho
        self.T_min, self.T_max = bounds_time

        #loss weights
        self.weights_inner, self.weights_lower, self.weights_upper, self.weights_terminal = boundary_loss_weights

        #parameters for PINN
        self.input_dim, self.hidden_dim, self.num_layers, self.batch_boundary, self.batch_interior = PINN_params

        self.net = self.FeedForwardStructure(self.input_dim, self.hidden_dim, self.num_layers)

        self.a_h = -0.5
        self.b_h = 1/self.T_max
        
        S = self.uniform_distribution(self.S_min, self.S_max, (self.batch_interior,))
        r = self.uniform_distribution(self.r_min, self.r_max, (self.batch_interior,))
        nu = self.uniform_distribution(self.nu_min, self.nu_max, (self.batch_interior,))
        
        self.a_S = -torch.mean(S)/torch.std(S)
        self.b_S = 1/torch.std(S)

        self.a_r = -torch.mean(r)/torch.std(r)
        self.b_r = 1/torch.std(r)

        self.a_nu = -torch.mean(nu)/torch.std(nu)
        self.b_nu = 1/torch.std(nu)

        # self.b_S = 12**0.5 / (self.S_max-self.S_min)
        # self.a_S = -0.5 * (self.S_max+self.S_min) * self.b_S

        # self.b_r = 12**0.5 / (self.r_max-self.r_min)
        # self.a_r = -0.5 * (self.r_max+self.r_min) * self.b_r
        
        # self.b_nu = 12**0.5 / (self.nu_max-self.nu_min)
        # self.a_nu = -0.5 * (self.nu_max+self.nu_min) * self.b_nu

    ### ------ Feed Foward Neural Netrowk ------ 
    class FeedForwardStructure(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, activation_func = nn.Tanh): #other activation funcs: nn.Tanh, nn.siLU
            super().__init__()

            self.input = nn.Linear(input_dim, hidden_dim)
            self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim+input_dim, hidden_dim) for _ in range(num_layers-2)])
            self.out = nn.Linear(hidden_dim, 1)
            self.activation_func = activation_func()
            self._init()
        
        def _init(self):
            # choose gain from the actual activation
            if isinstance(self.activation_func, nn.Tanh): gain = nn.init.calculate_gain('tanh')
            else: gain = 1.0 # SiLU/Swish etc. -> use 1.0

            nn.init.xavier_uniform_(self.input.weight, gain=gain)
            nn.init.zeros_(self.input.bias)
            for layer in self.hidden_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
                    nn.init.zeros_(layer.bias)

            nn.init.xavier_uniform_(self.out.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.out.bias)

        def forward(self, z):
            z_copy = z
            z_active = self.activation_func(self.input(z))
            for layer in self.hidden_layers:
                z_active = torch.cat([z_active, z_copy], dim=1)
                z_active = self.activation_func(layer(z_active))
            return self.out(z_active).squeeze(-1)
        
    ### ------ Helper functions for the term structure ------ 
    def psi_CIR(self, t):
        x0, kappa_r, theta_r, sigma_r = self.Theta_r
        f_Svensson = self.beta0 + self.beta1*torch.exp(-self.a1*t) + self.beta2*self.a1*t*torch.exp(-self.a1*t) + self.beta3*self.a2*t*torch.exp(-self.a2*t)
        
        h = self.h
        exp_term = torch.exp(t*h) - 1
        sum_term = kappa_r + h
        temp1 = 2*kappa_r*theta_r*exp_term / (2*h + sum_term*exp_term)
        temp2 = x0 * 4*h**2 * torch.exp(t*h) / (2*h + sum_term*exp_term)**2
        f_CIR = temp1 + temp2

        return f_Svensson - f_CIR
    
    def psi_CIR_deriv(self, t):
        x0, kappa_r, theta_r, sigma_r = self.Theta_r
        f_Svensson_deriv = -self.a1*self.beta1*torch.exp(-self.a1*t) + self.beta2*self.a1*torch.exp(-self.a1*t)*(1-self.a1*t) + self.beta3*self.a2*torch.exp(-self.a2*t)*(1-self.a2*t)
        
        h = self.h
        common_nominator = 4*h**2*torch.exp(t*h)
        common_denominator = (kappa_r + h)*torch.exp(t*h) - kappa_r + h
        f_CIR_deriv = kappa_r*theta_r*common_nominator/common_denominator**2 + x0 * common_nominator*h*((kappa_r+h)*torch.exp(t*h) + kappa_r - h)/common_denominator**3

        return f_Svensson_deriv - f_CIR_deriv
    
    def B(self, t, T):
            x0, kappa_r, theta_r, sigma_r = self.Theta_r
            numerator = 2*(torch.exp((T-t)*self.h)-1)
            denominator = 2*self.h + (kappa_r+self.h)*(torch.exp((T-t)*self.h)-1)
            return numerator/denominator
    
    def P(self, t, T, r):
        t, T, r = torch.broadcast_tensors(torch.as_tensor(t), torch.as_tensor(T), torch.as_tensor(r))
        
        x0, kappa_r, theta_r, sigma_r = self.Theta_r
        beta0, beta1, beta2, beta3, a1, a2 = [self.beta0, self.beta1, self.beta2, self.beta3, self.a1, self.a2]

        def y_Svensson(t, eps=1e-12):
            tau_safe = torch.where(t < eps, torch.full_like(t, eps), t)
            return beta0 + beta1*(1-torch.exp(-a1*tau_safe))/(a1*tau_safe) + beta2*((1-torch.exp(-a1*tau_safe))/(a1*tau_safe) - torch.exp(-a1*tau_safe)) + beta3*((1-torch.exp(-a2*tau_safe))/(a2*tau_safe) - torch.exp(-a2*tau_safe))
        
        def A(t, T):
            numerator = 2*self.h*torch.exp(0.5*(kappa_r+self.h)*(T-t))
            denominator = 2*self.h + (kappa_r+self.h)*(torch.exp((T-t)*self.h)-1)
            return (numerator/denominator)**(2*kappa_r*theta_r/sigma_r**2)
        
        def B(t, T):
            numerator = 2*(torch.exp((T-t)*self.h)-1)
            denominator = 2*self.h + (kappa_r+self.h)*(torch.exp((T-t)*self.h)-1)
            return numerator/denominator

        def A_hat(t, T):
            market_ZCB_price_t = torch.exp(-y_Svensson(t) * t)
            market_ZCB_price_T = torch.exp(-y_Svensson(T) * T)

            nominator = market_ZCB_price_T*A(0,t)*torch.exp(-B(0,t)*x0)*A(t,T)*torch.exp(B(t,T)*self.psi_CIR(t))
            denominator = market_ZCB_price_t*A(0,T)*torch.exp(-B(0,T)*x0)
            return nominator/denominator

        return A_hat(t,T) * torch.exp(-B(t,T)*r)

    ### ------ Sampling procedure ------ 
    def uniform_distribution(self, lower_bound, upper_bound, shape):
        return torch.empty(*shape, device=device).uniform_(lower_bound, upper_bound)
    
    def sample_interior(self, batch):
        T = self.uniform_distribution(self.T_min, self.T_max, (batch,))
        t = self.uniform_distribution(0, 1, (batch,))*T #ensures t ≤ T
        #t = torch.rand(batch, device=device)*T #ensures t ≤ T      

        S = self.uniform_distribution(self.S_min, self.S_max, (batch,))
        r = self.uniform_distribution(self.r_min, self.r_max, (batch,))
        nu = self.uniform_distribution(self.nu_min, self.nu_max, (batch,))

        kappa_nu = self.uniform_distribution(self.kappa_nu_min, self.kappa_nu_max, (batch,))
        theta_nu = self.uniform_distribution(self.theta_nu_min, self.theta_nu_max, (batch,))
        rho = self.uniform_distribution(self.rho_min, self.rho_max, (batch,))
        
        #sampling sigma_nu and ensuring sigma_nu^2 ≤ 2*kappa_nu*theta_nu
        sigma_nu = self.uniform_distribution(self.sigma_nu_min, self.sigma_nu_max, (batch,)) 
        sigma_nu = torch.where(sigma_nu**2 > 2*kappa_nu*theta_nu , torch.sqrt(2*kappa_nu*theta_nu), sigma_nu)

        return t, T, S, r, nu, kappa_nu, theta_nu, sigma_nu, rho

    def sample_upper_boundary(self, batch):
        T = self.uniform_distribution(self.T_min, self.T_max, (batch,))
        t = self.uniform_distribution(0, 1, (batch,))*T #ensures t ≤ T
        #t = torch.rand(batch, device=device)*T #ensures t ≤ T

        S = torch.full((batch,), self.S_max, device=device)
        r = self.uniform_distribution(self.r_min, self.r_max, (batch,))
        nu = self.uniform_distribution(self.nu_min, self.nu_max, (batch,))

        kappa_nu = self.uniform_distribution(self.kappa_nu_min, self.kappa_nu_max, (batch,))
        theta_nu = self.uniform_distribution(self.theta_nu_min, self.theta_nu_max, (batch,))
        rho = self.uniform_distribution(self.rho_min, self.rho_max, (batch,))
        
        #sampling sigma_nu and ensuring sigma_nu^2 ≤ 2*kappa_nu*theta_nu
        sigma_nu = self.uniform_distribution(self.sigma_nu_min, self.sigma_nu_max, (batch,)) 
        sigma_nu = torch.where(sigma_nu**2 >= 2*kappa_nu*theta_nu , torch.sqrt(2*kappa_nu*theta_nu), sigma_nu)

        return t, T, S, r, nu, kappa_nu, theta_nu, sigma_nu, rho

    def sample_lower_boundary(self, batch):
        T = self.uniform_distribution(self.T_min, self.T_max, (batch,))
        t = self.uniform_distribution(0, 1, (batch,))*T #ensures t ≤ T
        #t = torch.rand(batch, device=device)*T #ensures t ≤ T

        S = torch.full((batch,), self.S_min, device=device)
        r = self.uniform_distribution(self.r_min, self.r_max, (batch,))
        nu = self.uniform_distribution(self.nu_min, self.nu_max, (batch,))

        kappa_nu = self.uniform_distribution(self.kappa_nu_min, self.kappa_nu_max, (batch,))
        theta_nu = self.uniform_distribution(self.theta_nu_min, self.theta_nu_max, (batch,))
        rho = self.uniform_distribution(self.rho_min, self.rho_max, (batch,))

        #sampling sigma_nu and ensuring sigma_nu^2 ≤ 2*kappa_nu*theta_nu
        sigma_nu = self.uniform_distribution(self.sigma_nu_min, self.sigma_nu_max, (batch,)) 
        sigma_nu = torch.where(sigma_nu**2 >= 2*kappa_nu*theta_nu , torch.sqrt(2*kappa_nu*theta_nu), sigma_nu)

        return t, T, S, r, nu, kappa_nu, theta_nu, sigma_nu, rho
    
    def sample_terminal_condition(self, batch):
        T = self.uniform_distribution(self.T_min, self.T_max, (batch,))

        S = self.uniform_distribution(self.S_min, self.S_max, (batch,))
        r = self.uniform_distribution(self.r_min, self.r_max, (batch,))
        nu = self.uniform_distribution(self.nu_min, self.nu_max, (batch,))

        kappa_nu = self.uniform_distribution(self.kappa_nu_min, self.kappa_nu_max, (batch,))
        theta_nu = self.uniform_distribution(self.theta_nu_min, self.theta_nu_max, (batch,))
        rho = self.uniform_distribution(self.rho_min, self.rho_max, (batch,))
        
        #sampling sigma_nu and ensuring sigma_nu^2 ≤ 2*kappa_nu*theta_nu
        sigma_nu = self.uniform_distribution(self.sigma_nu_min, self.sigma_nu_max, (batch,)) 
        sigma_nu = torch.where(sigma_nu**2 >= 2*kappa_nu*theta_nu , torch.sqrt(2*kappa_nu*theta_nu), sigma_nu)

        return T, S, r, nu, kappa_nu, theta_nu, sigma_nu, rho

    # ------ Scaling variables and defining helper functions ------
    def scale_time(self, t):
        return self.a_h + self.b_h*t
    
    def scale_states(self, S, r, nu):
        S_tilde = self.a_S + self.b_S*S
        r_tilde = self.a_r + self.b_r*r
        nu_tilde = self.a_nu + self.b_nu*nu
        return S_tilde, r_tilde, nu_tilde

    def European_payoff_function(self, S):
        return torch.clamp(S - self.K, min=0)
    
    def forward_model_call(self, t_tilde, T_tilde, S_tilde, r_tilde, nu_tilde, kappa_nu, theta_nu, sigma_nu, rho):
        z = torch.stack([t_tilde, S_tilde, r_tilde, nu_tilde, kappa_nu, theta_nu, sigma_nu, rho, T_tilde], dim=-1)
        return self.net(z).squeeze(-1)

    # ------ Loss functions ------
    def loss_interior(self):
        t, T, S, r, nu, kappa_nu, theta_nu, sigma_nu, rho = self.sample_interior(self.batch_interior)
        x0, kappa_r, theta_r, sigma_r = self.Theta_r

        #scaling inputs 
        t_tilde = self.scale_time(t).requires_grad_(True)
        T_tilde = self.scale_time(T)
        S_tilde, r_tilde, nu_tilde = self.scale_states(S, r, nu)
        
        S_tilde = S_tilde.requires_grad_(True)
        r_tilde = r_tilde.requires_grad_(True)
        nu_tilde = nu_tilde.requires_grad_(True)

        V = self.forward_model_call(t_tilde=t_tilde, T_tilde=T_tilde, S_tilde=S_tilde, r_tilde=r_tilde, nu_tilde=nu_tilde, kappa_nu=kappa_nu, theta_nu=theta_nu, sigma_nu=sigma_nu, rho=rho)
        ones = torch.ones_like(V)
        
        #PDE residual
        partial_t_V = torch.autograd.grad(V, t_tilde, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        partial_S_V = torch.autograd.grad(V, S_tilde, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        partial_r_V = torch.autograd.grad(V, r_tilde, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        partial_nu_V = torch.autograd.grad(V, nu_tilde, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

        partial_SS_V = torch.autograd.grad(partial_S_V, S_tilde, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        partial_rr_V = torch.autograd.grad(partial_r_V, r_tilde, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        partial_nunu_V = torch.autograd.grad(partial_nu_V, nu_tilde, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        partial_Snu_V = torch.autograd.grad(partial_S_V, nu_tilde, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

        #helper variables
        psi = self.psi_CIR(t=t)
        psi_deriv = self.psi_CIR_deriv(t=t)

        mu_1 = (r_tilde-self.a_r)/self.b_r * (S_tilde-self.a_S)
        mu_2 = (kappa_r * (theta_r - (r_tilde-self.a_r)/self.b_r + psi) + psi_deriv) * self.b_r
        mu_3 = kappa_nu * (theta_nu - (nu_tilde-self.a_nu)/self.b_nu) * self.b_nu 

        diffusion1 = nu * (S_tilde-self.a_S)**2
        diffusion2 = (self.b_r * sigma_r)**2 * (r - psi)
        diffusion3 = self.b_nu * sigma_nu**2 * (nu_tilde - self.a_nu)
        cross_diffusion = 2*rho*sigma_nu * (nu_tilde-self.a_nu) * (S_tilde-self.a_S)

        residual = self.b_h*partial_t_V + mu_1*partial_S_V + mu_2*partial_r_V + mu_3*partial_nu_V + 0.5*(diffusion1*partial_SS_V + diffusion2*partial_rr_V + diffusion3*partial_nunu_V + cross_diffusion*partial_Snu_V) - V*r
        return torch.mean(residual**2)

    def loss_lower_boundary(self):
        t, T, S_min, r, nu, kappa_nu, theta_nu, sigma_nu, rho = self.sample_lower_boundary(self.batch_boundary)

        t_tilde = self.scale_time(t)
        T_tilde = self.scale_time(T)
        S_tilde, r_tilde, nu_tilde = self.scale_states(S_min, r, nu)
        S_tilde.requires_grad_(True)

        V_lower = self.forward_model_call(t_tilde=t_tilde, T_tilde=T_tilde, S_tilde=S_tilde, r_tilde=r_tilde, nu_tilde=nu_tilde, kappa_nu=kappa_nu, theta_nu=theta_nu, sigma_nu=sigma_nu, rho=rho)
        return torch.mean(V_lower**2)
    
    def loss_upper_boundary(self):
        t, T, S_max, r, nu, kappa_nu, theta_nu, sigma_nu, rho = self.sample_upper_boundary(self.batch_boundary)

        t_tilde = self.scale_time(t=t)
        T_tilde = self.scale_time(t=T)
        S_tilde, r_tilde, nu_tilde = self.scale_states(S_max, r, nu)
        S_tilde.requires_grad_(True)

        V_upper = self.forward_model_call(t_tilde=t_tilde, T_tilde=T_tilde, S_tilde=S_tilde, r_tilde=r_tilde, nu_tilde=nu_tilde, kappa_nu=kappa_nu, theta_nu=theta_nu, sigma_nu=sigma_nu, rho=rho)
        ones = torch.ones_like(V_upper)

        partial_S_V = torch.autograd.grad(V_upper, S_tilde, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        residual_upper = self.b_S * partial_S_V - 1
        return torch.mean(residual_upper**2)
    
    def loss_terminal(self):
        T, S, r, nu, kappa_nu, theta_nu, sigma_nu, rho = self.sample_terminal_condition(self.batch_boundary)

        T_tilde = self.scale_time(T)
        t_tilde = T_tilde
        S_tilde, r_tilde, nu_tilde = self.scale_states(S, r, nu)
        
        V_T = self.forward_model_call(t_tilde=t_tilde, T_tilde=T_tilde, S_tilde=S_tilde, r_tilde=r_tilde, nu_tilde=nu_tilde, kappa_nu=kappa_nu, theta_nu=theta_nu, sigma_nu=sigma_nu, rho=rho)
        payoff = self.European_payoff_function(S)

        residual_T = V_T - payoff
        return torch.mean(residual_T**2)

    def loss_total(self):
        loss_interior = self.weights_inner * self.loss_interior()
        loss_upper = self.weights_upper * self.loss_upper_boundary()
        loss_lower = self.weights_lower * self.loss_lower_boundary()
        loss_terminal = self.weights_terminal * self.loss_terminal()

        return loss_interior + loss_lower + loss_upper + loss_terminal, loss_interior, loss_upper, loss_lower, loss_terminal
    
    # ------ Pricing function ------
    def price_call(self, t, T, S, r, nu, kappa_nu, theta_nu, sigma_nu, rho):
        self.eval()
        dtype = next(self.parameters()).dtype
        dev = next(self.parameters()).device

        to_torch_tensor = lambda variable: torch.as_tensor(variable, dtype=dtype, device=dev).reshape(-1)
        t, T, S, r = to_torch_tensor(t), to_torch_tensor(T), to_torch_tensor(S), to_torch_tensor(r)
        kappa_nu, theta_nu, sigma_nu, rho, nu = to_torch_tensor(kappa_nu), to_torch_tensor(theta_nu), to_torch_tensor(sigma_nu), to_torch_tensor(rho), to_torch_tensor(nu)

        #scaling
        t_tilde = self.scale_time(t)
        T_tilde = self.scale_time(T)
        S_tilde, r_tilde, nu_tilde = self.scale_states(S, r, nu)

        with torch.no_grad():
            V = self.forward_model_call(t_tilde=t_tilde, T_tilde=T_tilde, S_tilde=S_tilde, r_tilde=r_tilde, nu_tilde=nu_tilde, kappa_nu=kappa_nu, theta_nu=theta_nu, sigma_nu=sigma_nu, rho=rho)
        return V.detach().cpu()

### ------ Training procedure ------
def train_PINN_ADAM(strike_price, bounds_S, bounds_r, Theta_r, Svensson_params, bounds_nu, bounds_rho, bounds_time, PINN_params, epochs_per_phase, learning_rates, boundary_loss_weights, print_every=50):
    model = PINN(strike_price=strike_price, bounds_S=bounds_S, Theta_r=Theta_r, bounds_r=bounds_r, bounds_nu=bounds_nu, bounds_rho=bounds_rho, bounds_time=bounds_time, Svensson_params=Svensson_params, PINN_params=PINN_params, boundary_loss_weights=boundary_loss_weights).to(device)
    total_phase_number = len(epochs_per_phase)

    history = []
    for i in range(total_phase_number):
        optimizer = optim.Adam(model.parameters(), lr=learning_rates[i])
        epochs = epochs_per_phase[i]

        for epoch in range(1, epochs+1):
            optimizer.zero_grad()
            total_loss, loss_interior, loss_upper, loss_lower, loss_terminal = model.loss_total()
            total_loss.backward()
            optimizer.step()

            if(epoch%print_every==0 or epoch in (1, epochs)):
                print(f'Phase {int(i+1)}/{int(total_phase_number)} | lr={learning_rates[i]} | {epoch}/{epochs} | Total loss={total_loss.item():.3f} | Interior loss={loss_interior.item():.3f} | Loss lower={loss_lower.item():.3f} | Loss upper={loss_upper.item():.3f} | Loss terminal={loss_terminal.item():.3f}')
            history.append(('ADAM', epoch, float(total_loss.detach()), float(loss_interior.detach()), float(loss_lower.detach()), float(loss_upper.detach()), float(loss_terminal.detach())))
    return model, history


### ------ Calibrate PINN ------
class CalibrationPINN:
    def __init__(self, PINN_model, df, t, S_0, r_0, ):
        self.t = t
        self.df = df
        self.S_0 = S_0
        self.r_0 = r_0

        self.PINN_model = PINN_model
        self.PINN_model.eval()
        
        #data preprocessing
        df = df.copy()

        self.T = pd.to_numeric(df['time_to_maturity_years'], errors='coerce').to_numpy(dtype=float)
        self.K = pd.to_numeric(df['strike'], errors='coerce').to_numpy(dtype=float)
        self.vega = pd.to_numeric(df['vega'], errors='coerce').to_numpy(dtype=float)
        self.C_market = pd.to_numeric(df['close'], errors='coerce').to_numpy(dtype=float)
        self.N = len(self.C_market)
    
    @staticmethod
    def unpack_parameters(Theta_nu):
        kappa_nu, theta_nu, sigma_nu, nu_t, rho = Theta_nu
        return float(kappa_nu), float(theta_nu), float(sigma_nu), float(nu_t), float(rho)
    
    def objective_func(self, Theta_nu, lambda_nu):
        BIG = 1e30
        try:
            kappa_nu, theta_nu, sigma_nu, nu_0, rho = self.unpack_parameters(Theta_nu)
        except Exception:
            return BIG
        if kappa_nu <= 0 or theta_nu <= 0 or sigma_nu <= 0:
            return BIG

        N = self.N
        model_K = np.full(N, self.PINN_model.K, dtype=float)
        
        t_vec = np.full(N, self.t, dtype=float)
        K_vec = self.K
        T_vec = self.T

        S_vec = np.full(N, self.S_0, dtype=float)
        r_vec = np.full(N, self.r_0, dtype=float)
        nu_vec = np.full(N, nu_0, dtype=float)

        kappa_vec = np.full(N, kappa_nu, dtype=float)
        theta_vec = np.full(N, theta_nu, dtype=float)
        sigma_vec = np.full(N, sigma_nu, dtype=float)
        rho_vec   = np.full(N, rho, dtype=float)

        with torch.no_grad():
            price_model = K_vec/model_K * self.PINN_model.price_call(t=t_vec, T=T_vec, S=S_vec*model_K/K_vec, r=r_vec, nu=nu_vec, kappa_nu=kappa_vec, theta_nu=theta_vec, sigma_nu=sigma_vec, rho=rho_vec).detach().cpu().numpy().reshape(-1)
        
        C_model = price_model
        C_market = self.C_market

        rel_error = ((C_model - C_market)/(C_market))**2
        
        return lambda_nu*np.mean(rel_error)

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

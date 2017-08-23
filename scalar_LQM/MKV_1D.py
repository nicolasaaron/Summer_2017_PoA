# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:53:14 2017

@author: iris
"""
#model description

# b(t,x,mu,a) = b1* x + bar_b1 * mu + b2 * a
# f(t,x,mu,a) = .5 * ( q * x^2 + bar_q *(x - s*mu)^2 + r* a^2)
# g(x,mu) = .5 * (qT * x^2 + bar_qT * (x - sT * mu)^2 )

#dynamic of X_t
# dX_t = b(t,x,mu,a)* t_step + sigma* dW_t

#%load_ext autoreload
#%autoreload 2

import numpy as np
import FBSDE_1D
import Riccati_1D
from scipy.stats import norm
import matplotlib.pyplot as plt
import time


#
def opt_control(X_t, eta,chi,const):
    return const * ( eta * X_t + chi );


#%%
#parameters for the dynamic X_t
# drift:
b1      = 1.
bar_b1  = 1.
b2      = 1.
# volatility
sigma   = 1.
# initial condition
X0      = 1.

# parameters for the cost functions
#running cost f
qt      = 1.
bar_qt  = 1.
st      = 1.
rt      = 1.
#terminal cost g
qT      = 1.
bar_qT  = 1.
sT      = 1.


# discretization parameters
T       = 1
t_step  = 0.001
t       = np.arange(0,T+t_step,t_step, dtype = 'float')
n_step  = len(t)

# %% find the mean function for distribution mu_t
# parameters for the FBSDE
a_t     = b1 + bar_b1
b_t     = - b2**2 / rt
c_t     = - (qt + bar_qt * (1. - st)) + st * bar_qt * (1. - st)
d_t     = - b1 - bar_b1
e_T     = qT + bar_qT * (1. - sT) - sT * bar_qT * (1. - sT)

bar_mu_t_MKV = FBSDE_1D.mean_fun(a_t,b_t,c_t,d_t,e_T,X0,T,t_step)

bar_eta_MKV = Riccati_1D.Riccati_1D((d_t-a_t)/2. , -b_t, -c_t, e_T, T, t)

#%%
plt.figure()
plt.plot(bar_mu_t_MKV)
plt.title('bar_mu')

plt.figure()
plt.plot(bar_eta_MKV)
plt.title('bar_eta')

# %% find the optimal strategy path coeffcient " eta_t" and "chi_t"
# parameters for the FBSDE
dyn_a_t = b1
dyn_b_t = - b2**2 / rt
dyn_c_t = bar_b1 * bar_mu_t_MKV
dyn_d_t = bar_qt * st * bar_mu_t_MKV + (st * bar_qt * (1. - st) - bar_b1 * bar_eta_MKV) * bar_mu_t_MKV
dyn_m_t = - (qt + bar_qt)

dyn_e_T = qT + bar_qT
dyn_tau_T = - bar_qT * sT * bar_mu_t_MKV[-1] - sT * bar_qT * (1. - sT) * bar_mu_t_MKV[-1]

control_coef_MKV = FBSDE_1D.control_fun(dyn_a_t, dyn_b_t, dyn_c_t, dyn_d_t, dyn_m_t,\
                                    dyn_e_T, dyn_tau_T,\
                                    T, t_step)

#%%
plt.figure()
plt.plot(control_coef_MKV['eta'])
plt.title('eta')

plt.figure()
plt.plot(control_coef_MKV['chi'])
plt.title('chi')

# %% simulate a trajectory of X_t
np.random.seed(seed = 0)

dW_t = norm.rvs(loc=0, scale=sigma * np.sqrt(t_step), size=n_step)

X_t = np.zeros(n_step,dtype=float)
X_t[0] = X0
for i in range(1,n_step,1):
    X_t[i] = X_t[i-1] + ((dyn_a_t + dyn_b_t *control_coef_MKV['eta'][i-1]) * X_t[i-1] \
                          + dyn_b_t * control_coef_MKV['chi'][i-1] + dyn_c_t[i-1]) * t_step \
             + dW_t[i-1]
             

alpha_t = opt_control(X_t,control_coef_MKV['eta'], control_coef_MKV['chi'], - b2 / rt )

# simulated running cost
f_t = .5 * (qt * X_t**2 + bar_qt * (X_t - st * bar_mu_t_MKV)**2 + rt * alpha_t**2)
# terminal cost
g_T   = .5 * (qT * X_t[-1]**2 + bar_qT * (X_t[-1] - sT * bar_mu_t_MKV[-1])**2 )
# sample value function in w.r.t. time
sample_v_t = (np.sum(f_t[:-1]) - np.insert(np.cumsum(f_t[:-1]),0,0) )*t_step + g_T
            

# express X_t as OU-process

tilde_Xt = np.zeros(n_step,dtype = float)
tilde_Xt[0] = X0
temp_sum = np.insert(np.cumsum( (dyn_a_t + dyn_b_t * bar_eta_MKV[:-1]) * t_step),0,0)
for i in range(1,n_step,1):
    tilde_Xt[i] = bar_mu_t_MKV[i] + np.sum( np.exp(temp_sum[i] - temp_sum[:(i-1)]) * dW_t[:(i-1)] )

#%% debug plot sample strategy
plt.figure()
plt.plot(np.cumsum(dW_t),color='r')
plt.title('brownian path')

plt.plot(X_t,color='blue')
plt.plot(tilde_Xt, color='green', linewidth = 0.5)


plt.figure()
plt.plot(alpha_t)
plt.title('alpha')

plt.figure()
plt.plot(f_t)
plt.title('f_t')

plt.figure()
plt.plot(sample_v_t)
plt.axhline(y=g_T,xmin=0,xmax=T,c="blue",linewidth=0.5,zorder=0)
plt.title('sample_v_t')
#%% compute the value function by simulation

N_simulate = 100000

v_t_sim_MKV = np.zeros(n_step)

#simulate bronian motions
dW_t = norm.rvs(loc=0, scale=sigma * np.sqrt(t_step), size=[N_simulate,n_step])

#simulate Xt
X_t = np.zeros([N_simulate,n_step],dtype=float)
X_t[:,0] = X0

for i in range(1,n_step,1):
    X_t[:,i] = X_t[:,i-1] + ((dyn_a_t + dyn_b_t *control_coef_MKV['eta'][i-1]) * X_t[:,i-1] \
                          + dyn_b_t * control_coef_MKV['chi'][i-1] + dyn_c_t[i-1]) * t_step \
            + dW_t[:,i-1]

#compute optimal strategy
alpha_t_MKV = opt_control(X_t,control_coef_MKV['eta'], control_coef_MKV['chi'], - b2 / rt )


# simulated running cost
f_t = .5 * (qt * X_t**2 + bar_qt * (X_t - st * bar_mu_t_MKV)**2 + rt * alpha_t_MKV**2)
# terminal cost
g_T   = .5 * (qT * X_t[:,-1]**2 + bar_qT * (X_t[:,-1] - sT * bar_mu_t_MKV[-1])**2 )
# sample value function in w.r.t. time
temp = np.insert(np.cumsum(f_t[:,:-1], axis=1),0,0, axis = 1)
sample_v_t =  (temp[:,-1].reshape(N_simulate,1) - temp)*t_step + g_T.reshape(N_simulate,1)

v_t_sim_MKV = np.sum(sample_v_t / N_simulate ,axis = 0)

#%% test dynamic process X_t
plt.figure()
plt.plot(np.sum(X_t/N_simulate,axis=0),color='blue')
plt.plot(bar_mu_t_MKV, color = 'red')
plt.title('bar_mu & sim_Xt_mu')

plt.figure()
plt.plot(t,v_t_sim_MKV)
plt.titel('v_t_sim_MKV')
#plt.axhline(y=v_0,xmin=0,xmax=1,c="blue",linewidth=0.5,zorder=0)


print('value function by simulation: {:.10f}'.format(v_t_sim_MKV[0]))
#print('value function by formula: {:.10f}'.format(v_0))

                   

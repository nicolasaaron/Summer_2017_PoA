# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:07:48 2017

@author: Zongjun
"""

import numpy as np
import Riccati_1D

def mean_fun(a,b,c,d,e,mu_0,T,t_step):
    t = np.arange(0,T+t_step,t_step, dtype ='float') 
    bar_eta = Riccati_1D.Riccati_1D((d-a)/2. , -b, -c, e, T, t)
    bar_mu = mu_0 * np.exp(a * t + b * np.insert(np.cumsum(bar_eta[:-1]),0,0.) * t_step )
    return bar_mu

def control_fun(a,b,c,d,m,e,tau, T,t_step):
    t = np.arange(0.,T+t_step, t_step, dtype = 'float')
    n_step = len(t)
    eta = Riccati_1D.Riccati_1D(-a, -b, -m, e, T, t)
    integral_eta= a * t + b * np.insert(np.cumsum(eta[:-1]),0,0) * t_step        
    chi = np.zeros(n_step,dtype='float')
    coef_temp = d - c * eta
   
    chi[n_step-1] = tau
    for i in range(0,n_step-1,1):
        chi[i] = tau * np.exp(integral_eta[-1] - integral_eta[i]) \
                 - np.sum(coef_temp[i:-1] * np.exp(integral_eta[(i+1):] - integral_eta[i])) * t_step
    
    ans = {'eta': eta, 'chi': chi}
    return ans;


def gamma_cst_fun(eta, chi, sigma, b,c,f,g,t_step):
    integral_t = .5 * np.cumsum(sigma**2 * eta[:-1] + b * chi[:-1]**2 + 2* c[:-1] * chi[:-1] + f[:-1] ) * t_step                       
    gamma = ( integral_t[-1] - np.insert(integral_t,0,0) ) + .5 * g                  
    return gamma
    
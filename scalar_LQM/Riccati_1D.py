# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:36:22 2017

@author: iris
"""
import numpy as np;

def Riccati_1D(A,B,C,gamma,T,t):
    # t = np.array(0,T,t_step)
    R = A**2 + B*C
    
    if R<0: 
        print("No solution")
        return;
    
    delta_p = - A + np.sqrt(R)
    delta_n = - A - np.sqrt(R)
    coef_1 = (delta_p - delta_n) * (T-t)

    return np.divide( - C * ( np.exp(coef_1) - 1) - gamma * (delta_p * np.exp(coef_1) - delta_n) \
                    , delta_n * np.exp(coef_1) - delta_p - gamma * B * (np.exp(coef_1) -1 ) );    
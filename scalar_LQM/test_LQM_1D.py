# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 01:53:35 2017

@author: iris
"""

"""
%load_ext autoreload
autoreload 2
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time


from LQM_1D import MFG_1D, MKV_1D

#%%
start = time.time()

test_MFG = MFG_1D()
test_MKV = MKV_1D()

#%%

params ={   'b1'      : 1.,
            'bar_b1'  : 1.,
            'b2'      : 1.,
            # volatility
            'sigma'   : 1.,
            # initial condition
            'X0'      : 1.,
            # parameters for the cost functions
            #running cost f
            'qt'      : 1.,
            'bar_qt'  : 1.,
            'st'      : 1.,
            'rt'      : 1.,
            #terminal cost g
            'qT'      : 0.,
            'bar_qT'  : 0.,
            'sT'      : 0.,
            # discretization parameters
            'T'       : 1,
            't_step'  : 0.001,
            # simulation
            'N_simulate' : 100000
        }

test_MFG.set_model_param(params)
test_MKV.set_model_param(params)


#%% test mean
test_MFG.mean_fun()
test_MKV.mean_fun()
"""
test_MFG.fig_bar_mu()
test_MKV.fig_bar_mu()
"""

# test control
test_MFG.control_fun()
test_MKV.control_fun()

"""
test_MFG.fig_control_coef()
test_MKV.fig_control_coef()
"""
"""
# test simulation
test_MFG.sample_simulation()
test_MKV.sample_simulation()
"""

#%% simulation with common BM
seed = 0
test_MFG.get_Brownian_Motion(seed)
#LQM_1D.fig_BM_sample()

BM_path = False
test_MFG.v_t_simulate(BM_path,seed=1)
test_MKV.v_t_simulate(BM_path,seed=1)

test_MFG.set_v_0_formula()

end = time.time()
print('elasps time {}'.format(end-start))



test_MFG.get_v_0_simulation()
test_MFG.get_v_0_formula()
test_MKV.get_v_0_simulation()

test_MFG.fig_simulation()
test_MKV.fig_simulation()


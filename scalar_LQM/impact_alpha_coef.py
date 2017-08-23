# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 04:21:52 2017

@author: iris
"""

import time
from LQM_1D import MFG_1D, MKV_1D
import numpy as np
import gc
import matplotlib.pyplot as plt
import pickle

#%%
def load_data(filename):
    
    try:
        with open(filename,'rb') as f:
            input_dict = pickle.load(f)
    except:
        input_dict = None
    return input_dict
    

#%%
param_name = 'rt'

min_rt = 0.002
max_rt = 1
step = 0.002
test_parameter = np.arange(min_rt, max_rt, step)
N_test = len(test_parameter)

list_MFG = []
list_MKV = []


seed =0
BM_path = False
Null_terminal = True

flag_t_step = False
t_step = 0.001

start = time.time()
for i in range(N_test):    
    print('iteration {:d}, test parameter equals {:.6f}'.format(i,test_parameter[i]))
    
    test_MFG=MFG_1D()
    test_MKV=MKV_1D()
    
    params = {param_name: test_parameter[i]}
    if Null_terminal:
        params.update({'qT':0., 'sT':0., 'bar_qT':0.})
    if flag_t_step:
        params.update({'t_step':t_step})
    test_MFG.set_model_param(params)
    test_MKV.set_model_param(params)
    
    test_MFG.get_Brownian_Motion(seed)
    
    test_MFG.mean_fun()
    test_MKV.mean_fun()
    
    test_MFG.control_fun()
    test_MKV.control_fun()   
    
    test_MFG.v_t_simulate()
    test_MKV.v_t_simulate()

    test_MFG.set_v_0_formula()

    end = time.time()
    test_MFG.get_v_0_simulation()
    test_MFG.get_v_0_formula()
    test_MKV.get_v_0_simulation()
    print('PoA = {:.10f}'.format(test_MFG.v_t_sim[0] / test_MKV.v_t_sim[0]))
    print('The elapsed time is {}\n'.format(end-start))
    
    list_MFG.append(test_MFG)
    list_MKV.append(test_MKV)
    
    del test_MFG
    del test_MKV
    #if (i % 10) == 0:
    #    gc.collect()

#%%
test_v_0_sim_MFG = np.empty(N_test)
test_v_0_sim_MKV = np.empty(N_test)
test_v_0_theo_MFG= np.empty(N_test)
test_PoA = np.empty(N_test)

for i in range(N_test):
    test_v_0_sim_MFG[i] = list_MFG[i].v_t_sim[0]
    test_v_0_theo_MFG[i] = list_MFG[i].v_0_formula
    test_v_0_sim_MKV[i] = list_MKV[i].v_t_sim[0]
    
    test_PoA[i] = test_v_0_sim_MFG[i] / test_v_0_sim_MKV[i]

#%%
plt.figure()
plt.plot(test_parameter,test_PoA)
plt.title('PoA')


#%%
flag_save_data = False
if flag_save_data:
    filename = 'PoA_'+param_name+'_0002_1.dat'
    data = {'param_name':param_name,
            'MFG':list_MFG,
            'MKV':list_MKV,
            'test_parameter':test_parameter,
            'test_v_0_sim_MFG':test_v_0_sim_MFG,
            'test_v_0_sim_MKV':test_v_0_sim_MKV,
            'test_v_0_theo_MFG':test_v_0_theo_MFG}
    with open(filename,'wb') as f:
        pickle.dump(data,f)
 

#%%

flag_load_data = False
if flag_load_data:
    #param_name = 'rt'
    filename = 'PoA_'+param_name+'_.dat'
    input_dict = load_data(filename)
    if input_dict is not None:
        list_MFG = input_dict['MFG']
        list_MKV = input_dict['MKV']
        test_parameter  = input_dict['test_parameter']
        test_v_0_sim_MFG = input_dict['test_v_0_sim_MFG']
        test_v_0_sim_MKV = input_dict['test_v_0_sim_MKV']
        test_v_0_theo_MFG = input_dict['test_v_0_theo_MFG']
    else:
        print('loading error')


#%%
plt.figure()
plt.plot(test_v_0_sim_MFG)
plt.title('MFG_simulation')

plt.figure()
plt.plot(test_v_0_sim_MKV)
plt.title('MKV_simulation')
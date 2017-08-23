# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:01:03 2017

@author: iris
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['interactive'] == True

def load_data(filename):
    
    try:
        with open(filename,'rb') as f:
            input_dict = pickle.load(f)
    except:
        input_dict = None
    return input_dict

#%%
param_name = 'b2'

filename = 'PoA_'+param_name+'_0002_1.dat'
input_dict_1 = load_data(filename)

filename = 'PoA_'+param_name+'_01_50.dat'
input_dict_2 = load_data(filename)

test_v_0_sim_MFG = np.concatenate((input_dict_1['test_v_0_sim_MFG'],input_dict_2['test_v_0_sim_MFG'][10:]))
test_v_0_sim_MKV = np.concatenate((input_dict_1['test_v_0_sim_MKV'],input_dict_2['test_v_0_sim_MKV'][10:]))
test_parameter  = np.concatenate((input_dict_1['test_parameter'],input_dict_2['test_parameter'][10:]))

test_PoA = test_v_0_sim_MFG / test_v_0_sim_MKV

plt.figure()
plt.semilogx(test_parameter, test_PoA)
plt.title('PoA')
plt.xlabel('b2')
plt.savefig('PoA_b2.jpeg')


#%%

param_name = 'rt'
filename = 'PoA_'+param_name+'_01_50.dat'
input_dict_3 = load_data(filename)

test_PoA = input_dict_3['test_v_0_sim_MFG'] / input_dict_3['test_v_0_sim_MKV']
test_parameter = input_dict_3['test_parameter']

plt.figure()
plt.plot(test_parameter, test_PoA)
plt.title('PoA')
plt.xlabel('rt')
plt.savefig('PoA_rt.jpeg')

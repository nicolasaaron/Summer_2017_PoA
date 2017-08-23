# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:14:52 2017

@author: iris
"""
import numpy as np
import FBSDE_1D
import Riccati_1D
from scipy.stats import norm
import matplotlib.pyplot as plt

class LQM_1D:
    
    dW_t = np.array([])
    
    def __init__(self):
        self.b1      = 1.
        self.bar_b1  = 1.
        self.b2      = 1.
        # volatility
        self.sigma   = 1.
        # initial condition
        self.X0      = 1.
        
        # parameters for the cost functions
        #running cost f
        self.qt      = 1.
        self.bar_qt  = 1.
        self.st      = 1.
        self.rt      = 1.
        #terminal cost g
        self.qT      = 1.
        self.bar_qT  = 1.
        self.sT      = 1.
        
        self.T       = 1.
        self.t_step  = 0.001
        self.t       = np.arange(0,self.T+self.t_step,self.t_step, dtype = 'float')
        self.n_step  = len(self.t) 
        
        # simulation parameters
        self.N_simulate = 100000
        
        
    def set_model_param(self, params):
        #print(params)
        if params is not None:
            self.b1     = params.get('b1',      self.b1)
            self.bar_b1 = params.get('bar_b1',  self.bar_b1)
            self.b2     = params.get('b2',      self.b2)
            self.sigma  = params.get('sigma',   self.sigma)
            self.X0     = params.get('X0',      self.X0)
            self.qt     = params.get('qt',      self.qt)
            self.bar_qt = params.get('bar_qt',  self.bar_qt)
            self.st     = params.get('st',      self.st)
            self.rt     = params.get('rt',      self.rt)
            self.qT     = params.get('qT',      self.qT)
            self.bar_qT = params.get('bar_qT',  self.bar_qT)
            self.sT     = params.get('sT',      self.sT)
            self.T      = params.get('T',       self.T)
            self.t_step = params.get('t_step',  self.t_step)
            self.N_simulate = params.get('N_simulate', self.N_simulate)
    
            if 'T' in params or 't_step' in params:
                self.t       = np.arange(0, self.T+self.t_step, self.t_step, dtype = 'float')
                self.n_step  = len(self.t)
                
    def set_FBSDE_param_mu(self):
        pass;
        
    def set_FBSDE_param_dyn(self):
        pass;
                
    def mean_fun(self):
        self.set_FBSDE_param_mu()
        self.bar_eta = Riccati_1D.Riccati_1D((self.d_t-self.a_t)/2. , -self.b_t, -self.c_t, self.e_T, self.T, self.t)
        self.bar_mu_t = self.X0 * np.exp(self.a_t * self.t + \
                                         self.b_t * np.insert(np.cumsum(self.bar_eta[:-1]),0,0.) * self.t_step )   

    def control_fun(self):
        self.set_FBSDE_param_dyn()
        temp_dict = FBSDE_1D.control_fun(self.dyn_a_t, self.dyn_b_t, self.dyn_c_t, self.dyn_d_t, self.dyn_m_t,\
                                    self.dyn_e_T, self.dyn_tau_T,\
                                    self.T, self.t_step)
        self.eta = temp_dict['eta']
        self.chi = temp_dict['chi']
    
    
    def opt_control(self, X_t, const):
        return const * ( self.eta * X_t + self.chi );
    
    def get_Brownian_Motion(self, *args):
        if args is not None:
            np.random.seed(seed = args[0])
        #simulate bronian motions
        LQM_1D.dW_t = norm.rvs(loc=0, scale=self.sigma * np.sqrt(self.t_step), size=[self.N_simulate,self.n_step])
       
       
    def v_t_simulate(self, BM_path = False, **kwargs):
        if BM_path:
            if kwargs is not None:
                np.random.seed(seed = kwargs.get('seed', 0))
            dW_t = norm.rvs(loc=0, scale=self.sigma * np.sqrt(self.t_step), size=[self.N_simulate,self.n_step])
        else:
            dW_t = LQM_1D.dW_t
        
        self.v_t_sim = np.zeros(self.n_step)
        
        X_t = np.zeros([self.N_simulate,self.n_step],dtype=float)
        X_t[:,0] = self.X0
        for i in range(1,self.n_step,1):
            X_t[:,i] = X_t[:,i-1] + ((self.dyn_a_t + self.dyn_b_t * self.eta[i-1]) * X_t[:,i-1] \
                                  + self.dyn_b_t * self.chi[i-1] + self.dyn_c_t[i-1]) * self.t_step \
                    + dW_t[:,i-1]
            
        alpha_t = self.opt_control(X_t, - self.b2 / self.rt )    
        # simulated running cost
        f_t = .5 * (self.qt * X_t**2 + self.bar_qt * (X_t - self.st * self.bar_mu_t)**2 + self.rt * alpha_t**2)
        # terminal cost
        g_T   = .5 * (self.qT * X_t[:,-1]**2 + self.bar_qT * (X_t[:,-1] - self.sT * self.bar_mu_t[-1])**2 )
        # sample value function in w.r.t. time
        temp = np.insert(np.cumsum(f_t[:,:-1], axis=1),0,0, axis = 1)
        sample_v_t =  (temp[:,-1].reshape(self.N_simulate,1) - temp)*self.t_step + g_T.reshape(self.N_simulate,1)
        
        self.v_t_sim = np.sum(sample_v_t / self.N_simulate ,axis = 0)
    
    def get_v_0_simulation(self):
        print('simulaiton: v(0,X0)={:.10f}'.format(self.v_t_sim[0])) 
    
    ##########################################################
    # plot figures
    ##########################################################
    def fig_bar_mu(self):
        plt.figure()
        plt.plot(self.bar_mu_t)
        plt.title('bar_mu_t')
        
        plt.figure()
        plt.plot(self.bar_eta)
        plt.title('bar_eta')
        
    def fig_control_coef(self):
        plt.figure()
        plt.plot(self.eta)
        plt.title('eta')
        
        plt.figure()
        plt.plot(self.chi)
        plt.title('chi')
        
    def fig_simulation(self):        
        plt.figure()
        plt.plot(self.t,self.v_t_sim)
       
    @classmethod
    def fig_BM_sample(cls):
        plt.figure()
        plt.plot(cls.dW_t[0,:])
        plt.title('dW_t[0]')
        
    ##################################################################
    # Debug: sample simulation
    ##################################################################
    def sample_simulation(self):
        np.random.seed(seed = 1)

        dW_t = norm.rvs(loc=0, scale=self.sigma * np.sqrt(self.t_step), size=self.n_step)
        X_t = np.zeros(self.n_step,dtype=float)
        X_t[0] = self.X0
        for i in range(1,self.n_step,1):
            X_t[i] = X_t[i-1] + ((self.dyn_a_t + self.dyn_b_t *self.eta[i-1]) * X_t[i-1] \
                                  + self.dyn_b_t * self.chi[i-1] + self.dyn_c_t[i-1]) * self.t_step \
                     + dW_t[i-1]
            
        alpha_t = self.opt_control(X_t, - self.b2 / self.rt )
        
        # simulated running cost
        f_t = .5 * (self.qt * X_t**2 + self.bar_qt * (X_t - self.st * self.bar_mu_t)**2 + self.rt * alpha_t**2)
        # terminal cost
        g_T   = .5 * (self.qT * X_t[-1]**2 + self.bar_qT * (X_t[-1] - self.sT * self.bar_mu_t[-1])**2 )
        # sample value function in w.r.t. time
        sample_v_t = (np.sum(f_t[:-1]) - np.insert(np.cumsum(f_t[:-1]),0,0) )*self.t_step + g_T
        
        plt.figure()
        plt.plot(np.cumsum(dW_t),color='r')
        plt.plot(X_t)
        plt.title('X_t & W_t')
        
        plt.figure()
        plt.plot(alpha_t)
        plt.title('alpha_t')
        
        plt.figure()
        plt.plot(f_t)
        plt.title('f_t')
        
        plt.figure()
        plt.plot(sample_v_t)
        plt.axhline(y=g_T,xmin=0,xmax=1,c="blue",linewidth=0.5,zorder=0)
        plt.title('sample_v_t')
                   
        
"""
MFG subclass
"""
class MFG_1D(LQM_1D):
    
    def set_FBSDE_param_mu(self):
        self.a_t     = self.b1 + self.bar_b1
        self.b_t     = - self.b2**2 / self.rt
        self.c_t     = - (self.qt + self.bar_qt * (1. - self.st))
        self.d_t     = - self.b1
        self.e_T     = self.qT + self.bar_qT * (1. - self.sT)
    
    def set_FBSDE_param_dyn(self):
        self.dyn_a_t = self.b1
        self.dyn_b_t = - self.b2**2 / self.rt
        self.dyn_c_t = self.bar_b1 * self.bar_mu_t
        self.dyn_d_t = self.bar_qt * self.st * self.bar_mu_t
        self.dyn_m_t = - (self.qt + self.bar_qt)
        self.dyn_f_t = self.bar_qt * self.st**2 * self.bar_mu_t**2
               
        self.dyn_g_T = self.bar_qT * self.sT**2 * self.bar_mu_t[-1]**2
        self.dyn_e_T = self.qT + self.bar_qT
        self.dyn_tau_T = - self.bar_qT * self.sT * self.bar_mu_t[-1]
    
    def control_fun(self):
        super(MFG_1D,self).control_fun()
        self.gamma = FBSDE_1D.gamma_cst_fun(self.eta, self.chi,\
                                            self.sigma, self.dyn_b_t, self.dyn_c_t, self.dyn_f_t,self.dyn_g_T,self.t_step)
        
    def set_v_0_formula(self):
        self.v_0_formula = .5 * self.X0**2 * self.eta[0] + self.chi[0] * self.X0 + self.gamma[0]
    
    def fig_control_coef(self):
        super(MFG_1D, self).fig_control_coef()
        
        plt.figure()
        plt.plot(self.gamma)
        plt.title('gamma')
        
    def fig_simulation(self):
        self.set_v_0_formula()        
        plt.figure()
        plt.plot(self.t,self.v_t_sim)
        plt.axhline(y=self.v_0_formula,xmin=0,xmax=1,c="blue",linewidth=0.5,zorder=0)
   
    def get_v_0_simulation(self):
        print('MFG simulation: v(0,X0)={:.10f}'.format(self.v_t_sim[0]))
    
    def get_v_0_formula(self):
        print('MFG formula: v(0,X0)={:.10f}'.format(self.v_0_formula)) 



"""
MKV subclass
"""
class MKV_1D(LQM_1D):
    
    def set_FBSDE_param_mu(self):
        self.a_t     = self.b1 + self.bar_b1
        self.b_t     = - self.b2**2 / self.rt
        self.c_t     = - (self.qt + self.bar_qt * (1. - self.st)) + self.st * self.bar_qt * (1. - self.st)
        self.d_t     = - self.b1 - self.bar_b1
        self.e_T     = self.qT + self.bar_qT * (1. - self.sT) - self.sT * self.bar_qT * (1. - self.sT)
        
    def set_FBSDE_param_dyn(self):         
        self.dyn_a_t = self.b1
        self.dyn_b_t = - self.b2**2 / self.rt
        self.dyn_c_t = self.bar_b1 * self.bar_mu_t
        self.dyn_d_t = self.bar_qt * self.st * self.bar_mu_t \
                        + (self.st * self.bar_qt * (1. - self.st) - self.bar_b1 * self.bar_eta) * self.bar_mu_t
        self.dyn_m_t = - (self.qt + self.bar_qt)
        
        self.dyn_e_T = self.qT + self.bar_qT
        self.dyn_tau_T = - self.bar_qT * self.sT * self.bar_mu_t[-1] \
                         - self.sT * self.bar_qT * (1. - self.sT) * self.bar_mu_t[-1]
     
    def get_v_0_simulation(self):
        print('MKV simulation: v(0,X0)={:.10f}'.format(self.v_t_sim[0]))          
        

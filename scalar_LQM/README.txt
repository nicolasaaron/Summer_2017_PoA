LQM_1D.py:  a class for scalar LQM model

Riccati_1D.py: a function solve the scalar Riccati equation

FBSDE_1D.py: functions gives back the solution of FBSDE, namely the mean function and the optimal strategy coefficients



test_MFG_1D.py : test the value function in MFG setting
test_MKV_1D.py : test the value function in MKV setting
test_LQM_1D.py : test the class LQM_1D

impact_alpha_coef.py	: compute the PoA w.r.t. different coefficients ('b2', and 'rt')
						  
						  results are saved into 
						  'PoA_XX_01_50.dat' files, the range of parameters is 0.1 -- 50, with step 0.1
						  'PoA_XX_0002_1.dat' files, the range of parameters is 0.002 -- 1, with step 0.002
						  
						  The simulation time range is [0,1] with step 0.001
						  
						  
	
figure_PoA: plot the figure of PoA

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 08:56:36 2022

@author: moor
"""

import numpy as np
from functools import partial #for multiprocessing
from multiprocessing import Pool #for multiprocessing
import random as rnd #for SSA 
from scipy.integrate import odeint #for numerical integrations
import scipy.integrate as intgr 
import matplotlib #for plotting
import matplotlib.pyplot as plt

from Trajectories import computeTrajectoryAB,parallelisationAB #algorithm to calculte the path mutual information of the two node network + its parallelisation
from functions import FF2Node,updateMatrixB,dN,variance #functions to support the algorithm
from exactsolutions import evolveQuasiExactAB #set of differential equations for the exact integration of the filtering equation for the three node network


const={"c1": 1,"c2":0.1,"c3":0.1,"c4":0.1} #reaction constants
const=list(const.values())
a0=10
b0=10
iniconds=[a0,b0] #initial conditions of the species
timevec=np.linspace(0,200,150) #integration time 
laenge=int(len(timevec)+1) #length of the final trajectories
core=1 #cores of the computer for multiprocessing
MC=1 #sample size for Monte Carlo average; lowering the sample size will speed up the calculation; for the exact calculation it is recommended to take a low sample size
dim=150 #dimension of the lattice for numerically integrating the filtering equation
#%% Calculation of the path mutual information, its rate and all the species-trajectories
if __name__ == '__main__': 
    exact=True #set False for integration of only the moment closure system. If exact=True, it is recommended to use lower sample size
    if exact==True:
        a_g,b_g,mean,secmom,mi,misq,rate,a1b,a2b,miclo,miclosq,rateclo=parallelisationAB(core,MC,exact,iniconds,const,dim,laenge,timevec)
    else:
        a_g,b_g,a1b,a2b,miclo,miclosq,rateclo=parallelisationAB(core,MC,exact,iniconds,const,dim,laenge,timevec)
        
    mivar=variance(miclo,miclosq) #variance of the path mutual information 
    ratevar=variance(rateclo,miclosq/(timevec*timevec)) #variance of the path mutual information rate   

#%% Reproducing the plot in the appendix, works only for exact==True
if __name__ == '__main__':
    varmomclos=variance(a1b,a2b)
    varexact=variance(mean,secmom)
    lwd=3
    matplotlib.rc('font',size=30)
    plt.rcParams['ps.useafm']=True
    plt.rcParams['axes.linewidth'] = 2
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['FreeSans']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(1,figsize=(10,10))
    plt.step(timevec,a_g,color='cornflowerblue',linewidth=lwd)
    plt.plot(timevec,a1b,color='magenta',linewidth=lwd)
    if exact:
        plt.plot(timevec,mean,color='seagreen',linewidth=lwd)
    plt.plot(timevec,a1b+varmomclos,'--',color='magenta',linewidth=lwd)
    plt.plot(timevec,a1b-varmomclos,'--',color='magenta',linewidth=lwd)
    if exact:
        plt.plot(timevec,mean+varexact,'--',color='seagreen',linewidth=lwd)
        plt.plot(timevec,mean-varexact,'--',color='seagreen',linewidth=lwd)
    plt.xlabel('Time t')
    plt.ylabel('Copy Number of A')
    if exact:
        plt.legend(('SSA','Exp. via Mom. Clos.', 'Exact Exp.'),loc='best')
    else:
        plt.legend(('SSA','Exp. via Mom. Clos.'),loc='best')
    plt.axis([0,70,0,23])
    plt.grid(linewidth=2)
    plt.tight_layout()
    plt.savefig('Comparing_momentclosure_with_quasiexact.png',dpi=250)
    plt.show()
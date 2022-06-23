#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 08:56:36 2022

@author: moor
"""

import sys
sys.path.append("/Users/moor/Documents/PhD/MIPaper")
import numpy as np
from functools import partial
import random as rnd
from scipy.integrate import odeint
import scipy.integrate as intgr
from functions import FF2Node,matrixB,dN,variance
from Trajectories import trajectoryAB,parallelAB #parallelAB still is to be defined 
from exactsolutions import dglAB
from multiprocessing import Pool
from analyticalfunctions import FF2_analytical,reacvel_2nodes,pmirate2node,gaussrate2node
import matplotlib
import matplotlib.pyplot as plt

const={"c1": 1,"c2":0.1,"c3":0.1,"c4":0.1}
const=list(const.values())
a0=10
b0=10
iniconds=[a0,b0]
dtmax=1e-4
timevec=np.linspace(0,200,150)
laenge=int(len(timevec)+1)
core=24
MC=1
dim=150
#%%
if __name__ == '__main__': #Calculation of the path mutual information, its rate and all the species-trajectories
    exact=True
    if exact==True:
        a_g,b_g,mean,secmom,mi,misq,rate,a1b,a2b,miclo,miclosq,rateclo=parallelAB(core,MC,exact,iniconds,const,dim,laenge,timevec)
    else:
        a_g,b_g,a1b,a2b,miclo,miclosq,rateclo=parallelAB(core,MC,exact,iniconds,const,dim,laenge,timevec)
        
#%%
if __name__ == '__main__': #Further calculations and analytical calculations 
    mivar=variance(miclo,miclosq)
    ratevar=variance(rateclo,miclosq[:-1]/(timevec*timevec))      

    y0=[iniconds[0],0,0]
    #c_prod_list=np.linspace(0,100,1000)
    #ratevA=reacvel_2nodes(c_prod_list,const.copy(),iniconds,timevec,'A') #for calculating the reaction velocity 
    #ratevB=reacvel_2nodes(c_prod_list,const.copy(),iniconds,timevec,'B')
        
    analyticalmi=odeint(FF2_analytical,y0,timevec, args=(const,))[:,-1]
    
    pmi2rate=pmirate2node(const)
    gauss2rate=gaussrate2node(const)

#%%
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
    plt.plot(timevec,mean,color='seagreen',linewidth=lwd)
    plt.plot(timevec,a1b+varmomclos,'--',color='magenta',linewidth=lwd)
    plt.plot(timevec,a1b-varmomclos,'--',color='magenta',linewidth=lwd)
    plt.plot(timevec,mean+varexact,'--',color='seagreen',linewidth=lwd)
    plt.plot(timevec,mean-varexact,'--',color='seagreen',linewidth=lwd)
    plt.xlabel('Time t')
    plt.ylabel('Copy Number of A')
    plt.legend(('SSA','Exp. via Mom. Clos.', 'Exact Exp.'),loc='best')
    plt.axis([0,70,0,23])
    plt.grid(linewidth=2)
    plt.tight_layout()
    # plt.savefig('Example.pdf',dpi=250)
    plt.show()
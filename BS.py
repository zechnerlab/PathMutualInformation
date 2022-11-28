#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:22:30 2022

@author: moor
"""

'''
This script is written to execute the pipeline for calculating the path mutual information for the three node feed forward network presented in the main text.
It will generate the plots the figure 3b-d in the main text and figures 3 and 4 of the appendix.  
'''

import numpy as np
from functools import partial #for multiprocessing
from multiprocessing import Pool #for multiprocessing
import random as rnd #for SSA
from scipy.integrate import odeint #for numerical integrations
import matplotlib #for plotting
import matplotlib.pyplot as plt

from BSfunctions import computeTrajectoryBS,parallelisationBS,evolveBS,MonteCarlo,bistableswitch #imports the main algorithm, its parallelisation and functions needed within the algorithm

MC=500 #sample size for Monte Carlo average (smaller sample size than used for the paper)

c0=100 #20
mu=0.7#feedback strength 
K=30 #Michaelis constant
eps=0.03    #offset
const={'k1': c0*c0*c0/(K*K*K+c0*c0*c0)*mu+eps, 'k2': 0.1, 'k3': 1, 'k4': 0.1, 'k5': 0.1, 'k6': 0.1} #reaction constants 
const=list(const.values())

a0=10 #3
b0=100 #20

iniconds=[a0,b0,c0] #initial conditions
params=[K,eps,mu] #collects the parameters for the feedback 
core=1 #cores of the computer for multiprocessing; to be adapted 

#%% Calculating and plotting the mutual information rate and the entropy rates
if __name__ == '__main__':
    directory='datafiles/BistableSwitch/' 
    timevec=np.linspace(0,10000,300) #integration time
    mulist=[0.05,0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5] #list containing the values of mu to which we determine the rate
    rate,rate_A,rate_C,var,var_A,var_C=[np.zeros(len(mulist)) for i in range(6)]
    
    for i in range(len(mulist)):
        mu=mulist[i]
        params=[K,eps,mu]
        atdata,btdata,ctdata,b1acdata,b2acdata,a1cdata,a2cdata,b1cdata,b2cdata,abcdata,b1adata,b2adata,c1adata,c2adata,miadata,micdata,midata=parallelisationBS(core,MC,iniconds,const,params,int(len(timevec)+1),timevec)
        mia,varmia=MonteCarlo(miadata,MC)
        mic,varmic=MonteCarlo(micdata,MC)
        mi,varmi=MonteCarlo(midata,MC) 
        rate[i]=mi[-2]/(timevec[-2]-timevec[250])
        rate_A[i]=mia[-2]/(timevec[-2]-timevec[250])
        rate_C[i]=mic[-2]/(timevec[-2]-timevec[250])
        var[i]=varmi[-2]/((timevec[-2]-timevec[250])*np.sqrt(MC))*2.5 #variances are scaled with the square root of the sample size times 2.5 
        var_A[i]=varmia[-2]/((timevec[-2]-timevec[250])*np.sqrt(MC))*2.5 #we let the system equilibrate to steady state and delete the transient at timevec[250]
        var_C[i]=varmic[-2]/((timevec[-2]-timevec[250])*np.sqrt(MC))*2.5
    
    ##Alternatively to the calculation before, this will open the data points generated for the paper; using this section will overwrite the arrays generated before
    ##Note that for obtaining this data, the transient for reaching the steady state has been manually cutted out which does not happen here 
    # directory='/datafiles/BistableSwitch/' 
    # rate=np.loadtxt(directory+'totalrate.out')
    # var=np.loadtxt(directory+'var_rate.out') #the variances have been calculated via the second moment and scaling with the square root of the sample size (5000) times 2.5 
    # rate_C=np.loadtxt(directory+'rate_c.out')
    # var_C=np.loadtxt(directory+'var_rate_c.out')
    # rate_A=np.loadtxt(directory+'rate_a.out')
    # var_A=np.loadtxt(directory+'var_rate_a.out')
        
    lwd=3
    v=0 #counting variable of the trajectory of the sample to represent
    matplotlib.rc('font',size=38)
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['ps.useafm']=True
    matplotlib.rc('font',**{'family':'serif','serif':['Times New Roman']})
    plt.rcParams['pdf.fonttype'] = 42  
    plt.figure(1,figsize=(13,10)) #Total mutual information rate 
    plt.errorbar(mulist,rate,xerr=None,yerr=var,color='blue',fmt='o',elinewidth=2.5,capsize=7,markersize=9)
    plt.fill_between(mulist,rate-var,rate+var,color='blue',alpha=0.1)
    plt.xlabel(r'Feedback Strength $\mu$')
    plt.ylabel(r'Mutual Information Rate i')
    plt.axis([0,1.5,0.046,0.053])
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.grid(linewidth=2)
    plt.vlines(0.35,0,0.06,color='grey',linestyle='dashed',alpha=0.8,linewidth=3.5)
    plt.tight_layout()
    plt.savefig('TotalRate.png',dpi=250)
    plt.show()

    plt.subplots(figsize=(14,10)) #total rate splitted in its transfer entropies
    color='seagreen'  
    plt.xlabel(r'Feedback Strength $\mu$')
    plt.ylabel(r'Transfer Entropy $h^{A\rightarrow C}$', color=color)
    plt.errorbar(mulist,rate_C,xerr=None,yerr=var_C,color=color,fmt='o',elinewidth=2.5,capsize=7,markersize=9)
    plt.fill_between(mulist,rate_C-var_C,rate_C+var_C,color=color,alpha=0.1)
    plt.tick_params(axis='y', labelcolor=color)
    plt.axis([0,1.5,0.04,0.05])
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.grid(linewidth=2)
    plt.twinx()
    color = 'purple'
    plt.ylabel(r'Transfer Entropy $h^{C\rightarrow A}$', color=color)
    plt.errorbar(mulist,rate_A,xerr=None,yerr=var_A,color=color,fmt='o',elinewidth=2.5,capsize=7,markersize=9)
    plt.fill_between(mulist,rate_A-var_A,rate_A+var_A,color=color,alpha=0.1)
    plt.tick_params(axis='y', labelcolor=color)
    plt.vlines(0.35,0,0.06,color='grey',linestyle='dashed',alpha=0.8,linewidth=3.5)
    plt.axis([0,1.5,0,0.005])
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.tight_layout()
    plt.savefig('TransferEntropies.png',dpi=250)
    plt.show()
#%%  Creating data and plotting the histograms for representation of the switching 
if __name__ == '__main__':
    mu_list=[0.05,0.7,1]
    timevec=np.linspace(0,10000,300)
    for i in range(len(mu_list)):
        params=[K,eps,mu_list[i]]
        if mu_list[i]==0.05:
            iniconds=[1,1,1]
            xmax=30
        if mu_list[i]==0.7:
            iniconds=[10,100,100]
            xmax=140
        if mu_list[i]==1:
            iniconds=[10,100,100]
            xmax=200
        const={'k1': c0*c0*c0/(K*K*K+c0*c0*c0)*mu_list[i]+eps, 'k2': 0.1, 'k3': 1, 'k4': 0.1, 'k5': 0.1, 'k6': 0.1}
        const=list(const.values())
        atdata,btdata,ctdata,b1acdata,b2acdata,a1cdata,a2cdata,b1cdata,b2cdata,abcdata,b1adata,b2adata,c1adata,c2adata,miadata,micdata,midata=parallelisationBS(core,MC,iniconds,const,params,int(len(timevec)+1),timevec)
        plt.figure(1,figsize=(10,10))
        plt.hist(np.transpose(ctdata)[-2],bins=14,facecolor='b',alpha=0.75,density='True',align='left')
        plt.xlabel('Copy Number of C')
        plt.ylabel('Relative Frequency')
        plt.xlim(-0.5,xmax)
        plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        plt.grid(linewidth=2)
        plt.tight_layout()
        plt.savefig('hist-mu_%d.png'%mu_list[i],dpi=250)
        plt.show()
#%% generating data and plotting of the bifurcation plot in the appendix
if __name__ == '__main__':
    y0=[10,100,100] #initial conditions for the numerical integration
    mu_list=np.linspace(0.00001,20,2000) #values for mu
    time=np.linspace(0,200,200) #integration time
    C_mu_10=np.zeros(len(mu_list)) #final values of the expected value of C
    for i in range(len(mu_list)):
        params=[K,eps,mu_list[i]] #assigning parameters
        C_mu_10[i]=odeint(bistableswitch,y0,time,args=(const,params) )[-1,2] #numerical integration 
        
    y0=[0,0,0]
    C_mu_0=np.zeros(len(mu_list))
    for i in range(len(mu_list)):
        params=[K,eps,mu_list[i]]
        C_mu_0[i]=odeint(bistableswitch,y0,time,args=(const,params) )[-1,2]
        
    matplotlib.rc('font',size=30)
    plt.rcParams['ps.useafm']=True
    plt.rcParams['axes.linewidth'] = 2
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['FreeSans']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(1,figsize=(10,10))
    plt.plot(mu_list,C_mu_0,'b',linewidth=2.5)
    plt.plot(mu_list,C_mu_10,'g',linewidth=2.5)
    plt.xlabel(r'Parameter $\mu$')
    plt.ylabel(r'Equilibrium Value of $c$')
    plt.axis([0,20,0,2000])
    plt.grid(linewidth=2)
    plt.tight_layout()
    plt.savefig('Bifurcation.png',dpi=250)
    plt.show()
    


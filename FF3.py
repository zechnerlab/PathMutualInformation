#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:37:01 2022

@author: moor
"""

'''
This script is written to execute the pipeline for calculating the path mutual information for the three node feed forward network presented in the main text.
It will generate the figures 1b+c and figure 2b-c in the main text and figure 1 in the appendix.  
'''

import numpy as np
from functools import partial #for multiprocessing
from multiprocessing import Pool #for multiprocessing
import random as rnd #for SSA
import scipy.integrate as intgr #for numerical integrations
from scipy.integrate import odeint 
import matplotlib #for plotting 
import matplotlib.pyplot as plt

from Trajectories import computeTrajectoryAC,parallelisationAC #algorithm to calculte the path mutual information of the three node network + its parallelisation 
from functions import evolveFF3Node,updateMatrixAC,updateMatrixC,mean2d,dN,dN2D,variance #functions to support the algorithm  
from exactsolutions import evolveQuasiExactAC #set of differential equations for the exact integration of the filtering equation for the three node network
from analyticalfunctions import evolveFF3_analytical,calculateReacvel_3nodes,gaussrate3node #analytical functions calculating with the three node network

from Trajectories import computeTrajectoryAB,parallelisationAB #algorithm to calculate the path mutual information of the two node network + its parallelisation
from functions import evolveFF2Node,updateMatrixB,dN,variance  #functions to support the algorithm
from exactsolutions import evolveQuasiExactAB #set of differential equations for the exact integration of the filtering equation for the two node network
from analyticalfunctions import gaussrate2node,evolveFF2_analytical #analytical functions calculating the gaussian and analytical rates of the two node network



MC=10000 #sample size for Monte Carlo average; min=5; lowering the sample size will speed up the calculation; for the exact calculation it is recommended to take a low sample size
dim=50 #dimension of the lattice for numerically integrating the filtering equation

const={"c1": 1,"c2":0.1,"c3":1,"c4":0.1,"c5":1,"c6":0.1} #reaction constants
const=list(const.values())
a0=10 
b0=100
c0=1000
iniconds=[a0,b0,c0] #initial conditions of the species 
timevec=np.linspace(0,200,150) #integration time 
laenge=int(len(timevec)+1) #length of the final trajectories
core=1 #cores of the computer for multiprocessing
#%%
if __name__ == '__main__': #Calculation of the path mutual information, its rate and all the species-trajectories
    exact=False #set True for integration of the filtering equation. If exact=True, it is recommended to use lower sample size
    if exact==True:
        a_g,b_g,meanac,meanc,mi,misq,rate,miexdata,b1ac,b1c,miclo,miclosq,rateclo,miclodata=parallelisationAC(core,MC,exact,iniconds,const,dim,laenge,timevec)
        mivar_exact=variance(mi,misq)
        ratevar_exact=variance(rate,misq/(timevec*timevec))
    else:
        a_g,b_g,b1ac,b1c,miclo,miclosq,rateclo,miclodata=parallelisationAC(core,MC,exact,iniconds,const,dim,laenge,timevec)
    mivar=variance(miclo,miclosq) #variance of the path mutual information
    ratevar=variance(rateclo,miclosq/(timevec*timevec)) #variance of the path mutual information rate 
    
    lwd=3
    matplotlib.rc('font',size=30)
    plt.rcParams['ps.useafm']=True
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['FreeSans']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['axes.linewidth'] = 2
    plt.figure(1,figsize=(10,10))
    plt.plot(timevec,miclo,'b',linewidth=lwd)
    plt.fill_between(timevec,miclo-mivar,miclo+mivar,alpha=0.3)
    for i in range(0,5):
        plt.plot(timevec,miclodata[i],'b',linewidth=2,alpha=0.3)
    plt.plot()
    plt.axis([0,150,-1,20])
    plt.xlabel(r"Time t")
    plt.ylabel(r"Mutual Information I")
    plt.grid(linewidth=2)
    plt.tight_layout()
    plt.savefig('Mifinal.pdf',dpi=250)
    plt.show()
    
    plt.figure(1,figsize=(10,10))
    plt.plot(timevec,rateclo,'b',linewidth=lwd)
    plt.fill_between(timevec,rateclo[:]-ratevar[:],rateclo[:]+ratevar[:],alpha=0.3)
    for i in range(0,5):
        plt.plot(timevec,miclodata[i]/timevec,'b',linewidth=2,alpha=0.3)
    plt.plot()
    plt.axis([0,150,-0.02,0.15])
    plt.xlabel(r"Time t")
    plt.ylabel(r"Mutual Information Rate i")
    plt.grid(linewidth=2)
    plt.tight_layout()
    plt.savefig('ratefinal.pdf',dpi=250)
    plt.show()
#%% Obtaining rate points numerically for different reaction velocities (figure 2) (Alternatively, our already generated data can be used)
if __name__ == '__main__': 
    a0=10 
    b0=10
    c0=100
    iniconds=[a0,b0,c0]
    exact=False
    
    #calculation for v_A: 
    const={"c1": 1,"c2":0.1,"c3":0.1,"c4":0.1,"c5":1,"c6":0.1} 
    const=list(const.values())
    if exact==True: #this will give the figure shown in the appendix. It is recommended to use a low sample size (e.g. n=300)
        velpoints=[0.5,0.75,1,2,3] #points of reaction velocity 
        rateA_exact=np.zeros(len(velpoints))
        rateA=np.zeros(len(velpoints))
        varA_exact=np.zeros(len(velpoints))
        varA=np.zeros(len(velpoints))
        for i in range(len(velpoints)):
            const[0]=velpoints[i]
            const[1]=velpoints[i]/a0
            a_g,b_g,meanac,meanc,mi,misq,rate,miexdata,b1ac,b1c,miclo,miclosq,rateclo,miclodata=parallelisationAC(core,MC,exact,iniconds,const,dim,laenge,timevec)
            rateA_exact[i]=rate[-1]
            rateA[i]=rateclo[-1]
            varA_exact[i]=np.sqrt(misq[-1]/(timevec[-1]*timevec[-1])-np.array(rateA_exact[i])*np.array(rateA_exact[i]))/np.sqrt(MC)*2.5
            varA[i]=np.sqrt(miclosq[-1]/(timevec[-1]*timevec[-1])-np.array(rateA[i])*np.array(rateA[i]))/np.sqrt(MC)*2.5 #variances are scaled with the square root of the sample size times 2.5
            np.savetxt('exact_rate_species_A.out', rateA_exact)
            np.savetxt('closed_rate_species_A.out', rateA)
            np.savetxt('var_exact_rate_species_A.out',varA_exact)
            np.savetxt('var_rate_species_A.out',varA)
    else: #This will give the figure in the main text 
        velpoints=[0.5,0.75,1,2,3,5,6,10,15] #points of reaction velocity
        rateA=np.zeros(len(velpoints))
        varA=np.zeros(len(velpoints))
        for i in range(len(velpoints)):
            const[0]=velpoints[i]
            const[1]=velpoints[i]/a0
            a_g,b_g,b1ac,b1c,miclo,miclosq,rateclo,miclodata=parallelisationAC(core,MC,exact,iniconds,const,dim,laenge,timevec)
            rateA[i]=rateclo[-1]
            varA[i]=np.sqrt(miclosq[-1]/(timevec[-1]*timevec[-1])-np.array(rateA[i])*np.array(rateA[i]))/np.sqrt(MC)*2.5
            np.savetxt('closed_rate_species_A.out', rateA)
            np.savetxt('var_rate_species_A.out',varA)
            
    #calculation for v_B:
    const={"c1": 1,"c2":0.1,"c3":0.1,"c4":0.1,"c5":1,"c6":0.1} 
    const=list(const.values())
    if exact==True:
        velpoints=[0.5,0.75,1,2,3]
        rateB_exact=np.zeros(len(velpoints))
        rateB=np.zeros(len(velpoints))
        varB_exact=np.zeros(len(velpoints))
        varB=np.zeros(len(velpoints))
        for i in range(len(velpoints)):
            const[2]=velpoints[i]
            const[3]=a0*velpoints[i]/b0
            a_g,b_g,meanac,meanc,mi,misq,rate,miexdata,b1ac,b1c,miclo,miclosq,rateclo,miclodata=parallelisationAC(core,MC,exact,iniconds,const,dim,laenge,timevec)
            rateB_exact[i]=rate[-1]
            rateB[i]=rateclo[-1]
            varB_exact[i]=np.sqrt(misq[-1]/(timevec[-1]*timevec[-1])-np.array(rateB_exact[i])*np.array(rateB_exact[i]))/np.sqrt(MC)*2.5
            varB[i]=np.sqrt(miclosq[-1]/(timevec[-1]*timevec[-1])-np.array(rateB[i])*np.array(rateB[i]))/np.sqrt(MC)*2.5
            np.savetxt('exact_rate_species_B.out', rateB_exact)
            np.savetxt('closed_rate_species_B.out', rateB)
            np.savetxt('var_exact_rate_species_B.out',varB_exact)
            np.savetxt('var_rate_species_B.out',varB)
    else:
        velpoints=[0.5,0.75,1,2,3,5,6,10,15]
        rateB=np.zeros(len(velpoints))
        varB=np.zeros(len(velpoints))
        for i in range(len(velpoints)):
            const[2]=velpoints[i]
            const[3]=a0*velpoints[i]/b0
            a_g,b_g,b1ac,b1c,miclo,miclosq,rateclo,miclodata=parallelisationAC(core,MC,exact,iniconds,const,dim,laenge,timevec)
            rateB[i]=rateclo[-1]
            varB[i]=np.sqrt(miclosq[-1]/(timevec[-1]*timevec[-1])-np.array(rateB[i])*np.array(rateB[i]))/np.sqrt(MC)*2.5
            np.savetxt('closed_rate_species_B.out', rateB)
            np.savetxt('var_rate_species_B.out',varB)
            
    #calculation for v_C:
    const={"c1": 1,"c2":0.1,"c3":0.1,"c4":0.1,"c5":1,"c6":0.1} 
    const=list(const.values())
    if exact==True:
        velpoints=[0.5,0.75,1,2,3]
        rateC_exact=np.zeros(len(velpoints))
        rateC=np.zeros(len(velpoints))
        varC_exact=np.zeros(len(velpoints))
        varC=np.zeros(len(velpoints))
        for i in range(len(velpoints)):
            const[4]=velpoints[i]
            const[5]=b0*velpoints[i]/c0
            a_g,b_g,meanac,meanc,mi,misq,rate,miexdata,b1ac,b1c,miclo,miclosq,rateclo,miclodata=parallelisationAC(core,MC,exact,iniconds,const,dim,laenge,timevec)
            rateC_exact[i]=rate[-1]
            rateC[i]=rateclo[-1]
            varC_exact[i]=np.sqrt(misq[-1]/(timevec[-1]*timevec[-1])-np.array(rateC_exact[i])*np.array(rateC_exact[i]))/np.sqrt(MC)*2.5
            varC[i]=np.sqrt(miclosq[-1]/(timevec[-1]*timevec[-1])-np.array(rateC[i])*np.array(rateC[i]))/np.sqrt(MC)*2.5
            np.savetxt('exact_rate_species_C.out', rateC_exact)
            np.savetxt('closed_rate_species_C.out', rateC)
            np.savetxt('var_exact_rate_species_C.out',varC_exact)
            np.savetxt('var_rate_species_C.out',varC)
    else:
        velpoints=[0.5,0.75,1,2,3,5,6,10,15]
        rateC=np.zeros(len(velpoints))
        varC=np.zeros(len(velpoints))
        for i in range(len(velpoints)):
            const[4]=velpoints[i]
            const[5]=b0*velpoints[i]/c0
            a_g,b_g,b1ac,b1c,miclo,miclosq,rateclo,miclodata=parallelisationAC(core,MC,exact,iniconds,const,dim,laenge,timevec)
            rateC[i]=rateclo[-1]
            varC[i]=np.sqrt(miclosq[-1]/(timevec[-1]*timevec[-1])-np.array(rateC[i])*np.array(rateC[i]))/np.sqrt(MC)*2.5
            np.savetxt('closed_rate_species_C.out', rateC)
            np.savetxt('var_rate_species_C.out',varC)
    
#%% Analytical calculations for the curves shown in figure 2 and the comparison to the gaussian mutual information 
if __name__ == '__main__':  
    
    y0=[iniconds[0],iniconds[1],0,0,0,0,0,0,0,0,0,0,0]
    sol=odeint(evolveFF3_analytical,y0,timevec, args=(const,)) #analytical calculation of the path mutual information corresponding to the numerical one 
    analyticalmi=sol[:,-1]
    analyticalrate=0.5*const[4]*(sol[:,5]-sol[:,11])/sol[:,1]
    
    const={"c1": 1,"c2":0.1,"c3":0.1,"c4":0.1,"c5":1,"c6":0.1}
    const=list(const.values())
    iniconds=[10,10,100]
    
    y0=[iniconds[0],iniconds[1],0,0,0,0,0,0,0,0,0,0,0]
    vel_list=np.linspace(0,100,1000)
    ratevA=calculateReacvel_3nodes(vel_list,const.copy(),iniconds,timevec,'A') #analytical calculation of the path mutual information rate dependent on the reaction velocities
    ratevB=calculateReacvel_3nodes(vel_list,const.copy(),iniconds,timevec,'B')
    ratevC=calculateReacvel_3nodes(vel_list,const.copy(),iniconds,timevec,'C')
    
    const_tempA=const.copy()
    const_tempC=const.copy()
    f3rateA=np.zeros(len(vel_list))
    f3rateC=np.zeros(len(vel_list)) 
    for i in range(len(vel_list)): 
        const_tempA[1]=vel_list[i]/10
        f3rateA[i]=intgr.quad(gaussrate3node, -np.inf, np.inf, args=(const_tempA,))[0] #gaussian mutual information rate depentend on v_A
        const_tempC[4]=vel_list[i]
        f3rateC[i]=intgr.quad(gaussrate3node, -np.inf, np.inf, args=(const_tempC,))[0] #gaussian mutual information rate depentend on v_C

#%% Generating the plots in figure 2b+c
if __name__ == '__main__': 
    ##Alternatively to the calculation before, this will open the data points generated for the paper; using this section will overwrite the arrays generated before the analytical calculations 
    # directory='datafiles/Feedforward/' #directory of the datafiles  
    # velpoints=[0.5,0.75,1,2,3,5,6,10,15,20,40]
    # rateA=np.loadtxt(directory+'rateA.out')   
    # rateB=np.loadtxt(directory+'rateB.out')
    # rateC=np.loadtxt(directory+'rateC.out')    
    # varA=np.loadtxt(directory+'varA.out') #the variances have been calculated via the second moment and scaling with the suare root of the sample size (n=10000) times 2.5
    # varB=np.loadtxt(directory+'varB.out')
    # varC=np.loadtxt(directory+'varC.out')
    # smplsz=600
    # rateA_exact=np.loadtxt(directory+'rateA_exact.out') 
    # rateB_exact=np.loadtxt(directory+'rateB_exact.out')
    # rateC_exact=np.loadtxt(directory+'rateC_exact.out')
    # varA_exact=np.loadtxt(directory+'varA_exact.out') #sample size n=600
    # varB_exact=np.loadtxt(directory+'varB_exact.out')
    # varC_exact=np.loadtxt(directory+'varC_exact.out')
    
    
    plt.figure(1,figsize=(10,10))
    plt.plot(vel_list,ratevA,'b',linewidth=lwd)
    plt.plot(vel_list,ratevB,'r', linewidth=lwd)
    plt.plot(vel_list,ratevC,'g', linewidth=lwd)
    plt.legend((r"$v_A$",r"$v_B$",r"$v_C$"),loc='best')
    plt.errorbar(velpoints,rateA,xerr=None,yerr=varA,fmt='bo',elinewidth=2.2,capsize=4,markersize=7)
    plt.errorbar(velpoints,rateB,xerr=None,yerr=varB,fmt='ro',elinewidth=2.2,capsize=4,markersize=7)
    plt.errorbar(velpoints,rateC,xerr=None,yerr=varC,fmt='go',elinewidth=1.2,capsize=4)
    plt.grid(linewidth=2)
    plt.axis([0,25,0,0.175])
    plt.xlabel(r"Relative Reaction Velocity $v_X$")
    plt.ylabel("Mutual Information Rate i")
    plt.tight_layout()
    plt.savefig('RRV.pdf', dpi=250)
    plt.show()
     
    plt.figure(1,figsize=(10,10))
    plt.plot(vel_list,ratevA,'b',linewidth=lwd)
    plt.plot(vel_list,f3rateA,'r',linewidth=lwd)
    plt.errorbar(velpoints,rateA,xerr=None,yerr=varA,fmt='bo',elinewidth=1.2,capsize=4)
    # plt.errorbar(velpoints[:5],rateA_exact,xerr=None,yerr=varA_exact,fmt='go',elinewidth=1.2,capsize=4) #for quasi-exact points
    plt.legend(("Analytical Rate","Gaussian Rate", 'Numerical Rate','Quasi-Exact Rate'),loc='best')
    plt.grid(linewidth=2)
    plt.axis([0,25,0,0.02])
    plt.xlabel(r"Relative Reaction Velocity $v_A$")
    plt.ylabel("Mutual Information Rate i")
    plt.tight_layout()
    plt.savefig('vA.pdf', dpi=250)
    plt.show()       
#%% Calculating the path mutual information of the two node network, comparing it to the gaussian mutual information and generating figure 2d
if __name__ == '__main__': 
    gaussrate,numexrate,varexact,analyticalrate_2node=[np.zeros(len(vel_list)) for i in range(4)]
    gaussrate2=gaussrate2node(const[:-2])
    
    const={"c1": 1,"c2":0.1,"c3":1,"c4":0.1}
    const=list(const.values())
    y0=[a0,0,0]
    sol=odeint(evolveFF2_analytical,y0,timevec, args=(const,))
    analytical_2node_rate=0.5*const[2]*sol[-1,1]/sol[-1,0]
    a0=10
    b0=100
    iniconds=[a0,b0]
    timevec=np.linspace(0,300,150)
    dim=150
    MC=1000 #our calculations have been performed with a sample size of n=10000, but a smaller one should be sufficient as well
    exact=True #we compare with the quasi-exact path mutual information of the two node network
    if exact==True:
        a_g,b_g,mean,secmom,mi,misq,rate,a1b,a2b,miclo,miclosq,rateclo=parallelisationAB(core,MC,exact,iniconds,const,dim,laenge,timevec)
    else:
        a_g,b_g,a1b,a2b,miclo,miclosq,rateclo=parallelisationAB(core,MC,exact,iniconds,const,dim,laenge,timevec)
    
    for i in range(len(vel_list)):
        gaussrate[i]=gaussrate2 #gaussian rate of the two state network
        analyticalrate_2node[i]=analytical_2node_rate
        numexrate[i]=rate[-2] #quasi-exact rate of the two state network; the values we obtained were rate[-2]=0.035453443149561 and secmom[-2]=2.246595566810135836e+02
        varexact[i]=np.sqrt(secmom[-2]/(timevec[-2]*timevec[-2])-rate[-2]*rate[-2])/np.sqrt(MC)*2.5 #variance of the quasi exact rate scaled with the sample size*2.5
    plt.figure(1,figsize=(10,10))
    plt.plot(vel_list,ratevC,'b',linewidth=lwd)
    plt.plot(vel_list,f3rateC,'r',linewidth=lwd)
    plt.plot(vel_list,numexrate,'--b',linewidth=lwd)
    plt.plot(vel_list, analyticalrate_2node, '--g', linewidth=lwd)
    plt.plot(vel_list,gaussrate,'--r',linewidth=lwd)
    # plt.errorbar(velpoints[:5],rateC_exact,xerr=None,yerr=varC_exact,fmt='go',elinewidth=1.2,capsize=4) #for quasi-exact points
    plt.legend(('Analyt. Approx.', 'Gaussian', 'Quasi-Exact (2 nodes)', 'Analyt. Approx. (2 nodes)', 'Gaussian (2 nodes)'),loc='best')
    plt.errorbar(velpoints,rateC,xerr=None,yerr=varC,fmt='bo',elinewidth=1.2,capsize=4,markersize=7)
    plt.fill_between(vel_list,numexrate-varexact,numexrate+varexact,color='blue',alpha=0.2)
    plt.grid(linewidth=2)
    plt.axis([0,25,0,0.04])
    plt.xlabel(r"Relative Reaction Velocity $v_C$")
    plt.ylabel("Mutual Information Rate i")
    plt.tight_layout()
    plt.savefig('CompbtwModels.pdf', dpi=250)
    plt.show()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:37:01 2022

@author: moor
"""

import sys
sys.path.append("/Users/moor/Documents/PhD/MIPaper")
import numpy as np
from functools import partial
import random as rnd
from scipy.integrate import odeint
import scipy.integrate as intgr
from functions import FF3Node,matrixAC,matrixC,mean2d,dN,dN2D,variance
from Trajectories import trajectoryAC,parallelAC
from exactsolutions import dglAC
from multiprocessing import Pool
from analyticalfunctions import FF3_analytical,reacvel_3nodes,pmirate3node,gaussrate3node
from analyticalfunctions import gaussrate2node

MC=4
dim=50

const={"c1": 1,"c2":0.1,"c3":1,"c4":0.1,"c5":1,"c6":0.1}
const=list(const.values())
a0=10
b0=100
c0=1000
iniconds=[a0,b0,c0]
timevec=np.linspace(0,200,150)
laenge=int(len(timevec)+1)
core=4

if __name__ == '__main__': #Calculation of the path mutual information, its rate and all the species-trajectories
    exact=False
    if exact==True:
        a_g,b_g,meanac,meanc,mi,misq,rate,miexdata,b1ac,b1c,miclo,miclosq,rateclo,miclodata=parallelAC(core,MC,exact,iniconds,const,dim,laenge,timevec)
    else:
        a_g,b_g,b1ac,b1c,miclo,miclosq,rateclo,miclodata=parallelAC(core,MC,exact,iniconds,const,dim,laenge,timevec)
#%%
if __name__ == '__main__': #Taking the data that already is saved
    directory='/Users/moor/Documents/PhD/MIPaper/datafiles/Feedforward/'
    miclodata=[]
    for i in range(0,5):
        miclodata.append(np.loadtxt(directory+'midata%d.out'%i))
    miclo=np.loadtxt(directory+'MIfinaldata.out')
    miclosq=np.loadtxt(directory+'MIsqfinaldata.out') #eventuell löschen 
    rateclo=np.loadtxt(directory+'Ratefinaldata.out')
#%%
if __name__ == '__main__': #Further calculations and analytical calculations 
    mivar=variance(miclo,miclosq)
    ratevar=variance(rateclo,miclosq[:-1]/(timevec*timevec))     
    
    y0=[iniconds[0],iniconds[1],0,0,0,0,0,0,0,0,0,0,0]
    sol=odeint(FF3_analytical,y0,timevec, args=(const,))
    analyticalmi=sol[:,-1]
    analyticalarate=0.5*const[4]*(sol[:,5]-sol[:,11])/sol[:,1]
    
    const={"c1": 1,"c2":0.1,"c3":0.1,"c4":0.1,"c5":1,"c6":0.1}
    const=list(const.values())
    iniconds=[10,10,100]
    
    y0=[iniconds[0],iniconds[1],0,0,0,0,0,0,0,0,0,0,0]
    c_prod_list=np.linspace(0,100,1000)
    ratevA=reacvel_3nodes(c_prod_list,const.copy(),iniconds,timevec,'A')
    ratevB=reacvel_3nodes(c_prod_list,const.copy(),iniconds,timevec,'B')
    ratevC=reacvel_3nodes(c_prod_list,const.copy(),iniconds,timevec,'C')
    
    pmirate=pmirate3node(const)
    
    const_tempA=const.copy()
    const_tempC=const.copy()
    f3rateA=np.zeros(len(c_prod_list))
    f3rateC=np.zeros(len(c_prod_list))
    pmi3=np.zeros(len(c_prod_list))
    for i in range(len(c_prod_list)):
        const_tempA[1]=c_prod_list[i]/10
        f3rateA[i]=intgr.quad(gaussrate3node, -np.inf, np.inf, args=(const_tempA,))[0]
        pmi3[i]=pmirate3node(const_tempA)
        const_tempC[4]=c_prod_list[i]
        f3rateC[i]=intgr.quad(gaussrate3node, -np.inf, np.inf, args=(const_tempC,))[0]

#%%
if __name__ == '__main__':
    directory='/Users/moor/Documents/PhD/MIPaper/datafiles/Feedforward/'
    cpoints=[0.5,0.75,1,2,3,5,6,10,15,20,40]
    rateA=np.loadtxt(directory+'rateA.out')   
    rateB=np.loadtxt(directory+'rateB.out')
    rateC=np.loadtxt(directory+'rateC.out')    
    varA=np.loadtxt(directory+'varA.out') #the variances have been calculated via the second moment and scaling with the sample size (10000) times 2.5
    varB=np.loadtxt(directory+'varB.out')
    varC=np.loadtxt(directory+'varC.out')
    smplsz=600
    rateA_exact=np.loadtxt(directory+'rateA_exact.out') 
    rateB_exact=np.loadtxt(directory+'rateB_exact.out')
    rateC_exact=np.loadtxt(directory+'rateC_exact.out')
    varA_exact=np.loadtxt(directory+'varA_exact.out') #sample size=600
    varB_exact=np.loadtxt(directory+'varB_exact.out')
    varC_exact=np.loadtxt(directory+'varC_exact.out')
    
    lwd=3
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rc('font',size=30)
    plt.rcParams['ps.useafm']=True
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['FreeSans']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['axes.linewidth'] = 2
    plt.figure(1,figsize=(10,10))
    plt.plot(timevec,miclo[:-1],'b',linewidth=lwd)
    plt.fill_between(timevec,miclo[:-1]-mivar[:-1],miclo[:-1]+mivar[:-1],alpha=0.3)
    for i in range(0,5):
        plt.plot(timevec,miclodata[i],'b',linewidth=2,alpha=0.3)
        # np.savetxt('midata%d.out'%i,midata[i][:-1])
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
    
    plt.figure(1,figsize=(10,10))
    plt.plot(c_prod_list,ratevA,'b',linewidth=lwd)
    plt.plot(c_prod_list,ratevB,'r', linewidth=lwd)
    plt.plot(c_prod_list,ratevC,'g', linewidth=lwd)
    plt.legend((r"$v_A$",r"$v_B$",r"$v_C$"),loc='best')
    plt.errorbar(cpoints,rateA,xerr=None,yerr=varA,fmt='bo',elinewidth=2.2,capsize=4,markersize=7)
    plt.errorbar(cpoints,rateB,xerr=None,yerr=varB,fmt='ro',elinewidth=2.2,capsize=4,markersize=7)
    plt.errorbar(cpoints,rateC,xerr=None,yerr=varC,fmt='go',elinewidth=1.2,capsize=4)
    plt.grid(linewidth=2)
    plt.axis([0,25,0,0.175])
    plt.xlabel(r"Relative Reaction Velocity $v_X$")
    plt.ylabel("Mutual Information Rate i")
    plt.tight_layout()
    # plt.savefig('RRV.pdf', dpi=250)
    plt.show()
     
    plt.figure(1,figsize=(10,10))
    plt.plot(c_prod_list,ratevA,'b',linewidth=lwd)
    plt.plot(c_prod_list,f3rateA,'r',linewidth=lwd)
    plt.errorbar(cpoints,rateA,xerr=None,yerr=varA,fmt='bo',elinewidth=1.2,capsize=4)
    # plt.errorbar(cpoints[:5],rateA_exact,xerr=None,yerr=varA_exact,fmt='go',elinewidth=1.2,capsize=4) #for quasi-exact points
    plt.legend(("Analytical Rate","Gaussian Rate", 'Numerical Rate','Quasi-Exact Rate'),loc='best')
    plt.grid(linewidth=2)
    plt.axis([0,25,0,0.02])
    plt.xlabel(r"Relative Reaction Velocity $v_A$")
    plt.ylabel("Mutual Information Rate i")
    plt.tight_layout()
    # plt.savefig('vA.pdf', dpi=250)
    plt.show()       
    
    gaussrate,numexrate=[np.zeros(len(c_prod_list)) for i in range(2)]
    gauss2rate=gaussrate2node(const[:-2])
    for i in range(len(c_prod_list)):
        gaussrate[i]=gauss2rate #klein
        # gaussrate[i]=0.11583123951777001
        numexrate[i]=0.035453443149561 #klein
        # numexrate[i]=0.17664999409502877
    plt.figure(1,figsize=(10,10))
    plt.plot(c_prod_list,ratevC,'b',linewidth=lwd)
    plt.plot(c_prod_list,f3rateC,'r',linewidth=lwd)
    plt.plot(c_prod_list,numexrate,'--b',linewidth=lwd)
    plt.plot(c_prod_list,gaussrate,'--r',linewidth=lwd)
    plt.errorbar(cpoints,rateC,xerr=None,yerr=varC,fmt='bo',elinewidth=1.2,capsize=4,markersize=7)
    # plt.errorbar(cpoints[:5],rateC_exact,xerr=None,yerr=varC_exact,fmt='go',elinewidth=1.2,capsize=4) #for quasi-exact points
    plt.legend(('Analyt. PMI', 'GMI - 3 nodes', 'Num. PMI - 2 nodes', 'GMI - 2 nodes','Clos. PMI - 3 nodes','Q.-E. PMI - 3 nodes'),loc='best')
    plt.grid(linewidth=2)
    plt.axis([0,25,0,0.05])
    plt.xlabel(r"Relative Reaction Velocity $v_C$")
    plt.ylabel("Mutual Information Rate i")
    plt.tight_layout()
    # plt.savefig('CompbtwModels.pdf', dpi=250)
    plt.show()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
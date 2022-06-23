#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:22:30 2022

@author: moor
"""
import sys
sys.path.append("/Users/moor/Documents/PhD/MIPaper")
import numpy as np
from functools import partial
import random as rnd
from scipy.integrate import odeint
from BSfunctions import parallelBS,BS,trajectoryBS,MonteCarlo,bistableswitch
import matplotlib
import matplotlib.pyplot as plt

MC=10000

c0=100
mu=0.55
K=30
eps=0.03   
const={'k1': c0*c0*c0/(K*K*K+c0*c0*c0)*mu+eps, 'k2': 0.1, 'k3': 1, 'k4': 0.1, 'k5': 0.1, 'k6': 0.1}
const=list(const.values())

a0=10
b0=100

iniconds=[a0,b0,c0]
params=[K,eps,mu]
timevec=np.linspace(0,150,150)
laenge=int(len(timevec)+1)
core=4 #for multiprocessing 

if __name__ == '__main__': #integration of conditional moments and mutual information/transfer entropy 
    atdata,btdata,ctdata,b1acdata,b2acdata,a1cdata,a2cdata,b1cdata,b2cdata,abcdata,b1adata,b2adata,c1adata,c2adata,miadata,micdata,midata=parallelBS(core,MC,iniconds,const,params,laenge,timevec)
    
    
    # np.savetxt('atdata_mu=055.out',atdata)
    # np.savetxt('btdata_mu=055.out',btdata)
    # np.savetxt('ctdata_mu=055.out',ctdata)
    # np.savetxt('b1acdata_mu=055.out',b1acdata)
    # np.savetxt('b2acdata_mu=055.out',b2acdata)
    # np.savetxt('a1cdata_mu=055.out',a1cdata)
    # np.savetxt('a2cdata_mu=055.out',a2cdata)
    # np.savetxt('b1cdata_mu=055.out',b1cdata)
    # np.savetxt('b2cdata_mu=055.out',b2cdata)
    # np.savetxt('abcdata_mu=055.out',abcdata)
    # np.savetxt('b1adata_mu=055.out',b1adata)
    # np.savetxt('b2adata_mu=055.out',b2adata)
    # np.savetxt('c1adata_mu=055.out',c1adata)
    # np.savetxt('c2adata_mu=055.out',c2adata)
    # np.savetxt('miadata_mu=055.out',miadata)
    # np.savetxt('micdata_mu=055.out',micdata)
    # np.savetxt('midata_mu=055.out',midata)
#%%
if __name__ == '__main__':     #Variances 
    at,varat=MonteCarlo(atdata,MC)
    bt,varbt=MonteCarlo(btdata,MC)
    ct,varct=MonteCarlo(ctdata,MC)
    bac,varbac=MonteCarlo(b1acdata,MC)
    ac,varac=MonteCarlo(a1cdata,MC)
    bc,varbc=MonteCarlo(b1cdata,MC)
    ba,varba=MonteCarlo(b1adata,MC)
    ca,varca=MonteCarlo(c1adata,MC)
    mia,varmia=MonteCarlo(miadata,MC)
    mic,varmic=MonteCarlo(micdata,MC)
    mi,varmi=MonteCarlo(midata,MC)    
#%% Plots 
if __name__ == '__main__':
    lwd=3
    v=0 #counting variable of the trajectory of the sample to represent
    matplotlib.rc('font',size=38)
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['ps.useafm']=True
    matplotlib.rc('font',**{'family':'serif','serif':['Times New Roman']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(1,figsize=(10,10)) #Single trajectory B conditional on A and C
    plt.step(timevec,btdata[v],'r',linewidth=lwd)
    plt.plot(timevec,b1acdata[v],'b',linewidth=lwd)
    plt.fill_between(timevec,b1acdata[v]-np.sqrt(b2acdata[v]-b1acdata[v]*b1acdata[v]),b1acdata[v]+np.sqrt(b2acdata[v]-b1acdata[v]*b1acdata[v]),alpha=0.3)
    plt.axis([0,150,50,190])
    plt.xlabel('Time t')
    plt.ylabel(r'Copy Number of B')
    plt.legend(('SSA',r'$\mathbb{E}[B|A_0^t, C_0^t]$'), loc='upper right')
    plt.grid(linewidth=2)
    plt.tight_layout()
    plt.savefig('B_AC.pdf',dpi=250)
    plt.show()
    plt.figure(1,figsize=(10,10)) #Single trajectory B conditional on A
    plt.step(timevec,btdata[v],'r',linewidth=lwd) 
    plt.plot(timevec,b1adata[v],'b',linewidth=lwd)
    plt.fill_between(timevec,b1adata[v]-np.sqrt(b2adata[v]-b1adata[v]*b1adata[v]),b1adata[v]+np.sqrt(b2adata[v]-b1adata[v]*b1adata[v]),alpha=0.3)
    plt.axis([0,150,50,190])
    plt.xlabel('Time t')
    plt.ylabel(r'Copy Number of B')
    plt.legend(('SSA',r'$\mathbb{E}[B|A_0^t]$'), loc='upper right')
    plt.grid(linewidth=2)
    plt.tight_layout()
    plt.savefig('B_A.pdf',dpi=250)
    plt.show()
    plt.figure(1,figsize=(10,10)) #Single trajectory B conditional on  C
    plt.step(timevec,btdata[v],'r',linewidth=lwd)
    plt.plot(timevec,b1cdata[v],'b',linewidth=lwd)
    plt.fill_between(timevec,b1cdata[v]-np.sqrt(b2cdata[v]-b1cdata[v]*b1cdata[v]),b1cdata[v]+np.sqrt(b2cdata[v]-b1cdata[v]*b1cdata[v]),alpha=0.3)
    plt.axis([0,150,50,190])
    plt.xlabel('Time t')
    plt.ylabel(r'Copy Number of B')
    plt.legend(('SSA',r'$\mathbb{E}[B|C_0^t]$'), loc='upper right')
    plt.grid(linewidth=2)
    plt.tight_layout()
    plt.savefig('B_C.pdf',dpi=250)
    plt.show()   
    plt.figure(1,figsize=(10,10)) #Averaged trajectories of B
    plt.plot(timevec,bt,'r',linewidth=lwd)
    plt.fill_between(timevec,bt-varbt,bt+varbt,alpha=0.3)
    plt.plot(timevec,bac,'b',linewidth=lwd)
    plt.plot(timevec,bc,'g',linewidth=lwd)
    plt.plot(timevec,ba,'m',linewidth=lwd)
    for i in range(v,v+5):
        plt.plot(timevec,b1acdata[i],'b',alpha=0.5)
    for i in range(v,v+5):
        plt.plot(timevec,b1cdata[i],'g',alpha=0.5)
    for i in range(v,v+5):
        plt.plot(timevec,b1adata[i],'m',alpha=0.5)
    plt.axis([0,150,50,190])
    plt.xlabel('Time t')
    plt.ylabel(r'Mean of B')
    plt.legend(('SSA',r'$\mathbb{E}[B|A_0^t, C_0^t]$',r'$\mathbb{E}[B|C_0^t]$',r'$\mathbb{E}[B|A_0^t]$'), loc='upper right')
    plt.grid(linewidth=2)
    plt.tight_layout()
    plt.savefig('B_average.pdf',dpi=250)
    plt.show() 
    
    plt.figure(1,figsize=(10,10)) #Single trajectory A conditional on C
    plt.step(timevec,atdata[v],'r',linewidth=lwd)
    plt.plot(timevec,a1cdata[v],'b',linewidth=lwd)
    plt.fill_between(timevec,a1cdata[v]-np.sqrt(a2cdata[v]-a1cdata[v]*a1cdata[v]),a1cdata[v]+np.sqrt(a2cdata[v]-a1cdata[v]*a1cdata[v]),alpha=0.3)
    plt.axis([0,150,0,25])
    plt.xlabel('Time t')
    plt.ylabel(r'Copy Number of A')
    plt.legend(('SSA',r'$\mathbb{E}[A|C_0^t]$'), loc='upper right')
    plt.grid(linewidth=2)
    plt.tight_layout()
    plt.savefig('A_C.pdf',dpi=250)
    plt.show()
    
    plt.figure(1,figsize=(10,10)) #Averaged trajectories of A conditional on C
    plt.plot(timevec,at,'r',linewidth=lwd)
    plt.fill_between(timevec,at-varat,at+varat,alpha=0.3)
    plt.plot(timevec,ac,'b',linewidth=lwd)
    for i in range(v,v+5):
        plt.plot(timevec,a1cdata[i],'b',alpha=0.5)
    plt.axis([0,150,0,20])
    plt.xlabel('Time t')
    plt.ylabel(r'Mean of A')
    plt.legend(('SSA',r'$\mathbb{E}[A|C_0^t]$'), loc='upper right')
    plt.grid(linewidth=2)
    plt.tight_layout()
    plt.savefig('A_average.pdf',dpi=250)
    plt.show() 
    
    plt.figure(1,figsize=(10,10)) #Single trajectory C conditional on A
    plt.step(timevec,ctdata[v],'r',linewidth=lwd)
    plt.plot(timevec,c1adata[v],'b',linewidth=lwd)
    plt.fill_between(timevec,c1adata[v]-np.sqrt(c2adata[v]-c1adata[v]*c1adata[v]),c1adata[v]+np.sqrt(c2adata[v]-c1adata[v]*c1adata[v]),alpha=0.3)
    plt.axis([0,150,60,170])
    plt.xlabel('Time t')
    plt.ylabel(r'Copy Number of C')
    plt.legend(('SSA',r'$\mathbb{E}[C|A_0^t]$'), loc='upper right')
    plt.grid(linewidth=2)
    plt.tight_layout()
    plt.savefig('C_A.pdf',dpi=250)
    plt.show()
    
    plt.figure(1,figsize=(10,10)) #Averaged trajectories of C conditional on A
    plt.plot(timevec,ct,'r',linewidth=lwd)
    plt.fill_between(timevec,ct-varct,ct+varct,alpha=0.3)
    plt.plot(timevec,ca,'b',linewidth=lwd)
    for i in range(v,v+5):
        plt.plot(timevec,c1adata[i],'b',alpha=0.5)
    plt.axis([0,150,60,170])
    plt.xlabel('Time t')
    plt.ylabel(r'Mean of C')
    plt.legend(('SSA',r'$\mathbb{E}[C|A_0^t]$'), loc='upper right')
    plt.grid(linewidth=2)
    plt.tight_layout()
    plt.savefig('C_average.pdf',dpi=250)
    plt.show()
#%%
    directory='/Users/moor/Documents/PhD/MIPaper/datafiles/BistableSwitch/'
    mulist=np.loadtxt(directory+'mulist.out')
    rate=np.loadtxt(directory+'totalrate.out')
    var=np.loadtxt(directory+'var_rate.out') #the variances have been calculated via the second moment and scaling with the sample size (100) times 2.5 
    ratec=np.loadtxt(directory+'rate_c.out')
    varc=np.loadtxt(directory+'var_rate_c.out')
    ratea=np.loadtxt(directory+'rate_a.out')
    vara=np.loadtxt(directory+'var_rate_a.out')
    
    plt.figure(1,figsize=(13,10)) #Total mutual information rate 
    plt.errorbar(mulist,rate,xerr=None,yerr=var,color='blue',fmt='o',elinewidth=2.5,capsize=7,markersize=9)
    plt.fill_between(mulist,rate-var,rate+var,color='blue',alpha=0.1)
    plt.xlabel(r'Feedback Strength $\mu$')
    plt.ylabel(r'Mutual Information Rate i')
    plt.axis([0,1.5,0.045,0.055])
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.grid(linewidth=2)
    plt.vlines(0.35,0,0.06,color='grey',linestyle='dashed',alpha=0.8,linewidth=3.5)
    plt.tight_layout()
    plt.savefig('TotalRate.pdf',dpi=250)
    plt.show()

    plt.subplots(figsize=(14,10)) #total rate splitted in its transfer entropies
    color='seagreen'  
    plt.xlabel(r'Feedback Strength $\mu$')
    plt.ylabel(r'Transfer Entropy $h^{A\rightarrow C}$', color=color)
    plt.errorbar(mulist,ratec,xerr=None,yerr=varc,color=color,fmt='o',elinewidth=2.5,capsize=7,markersize=9)
    plt.fill_between(mulist,ratec-varc,ratec+varc,color=color,alpha=0.1)
    plt.tick_params(axis='y', labelcolor=color)
    plt.axis([0,1.5,0.04,0.05])
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.grid(linewidth=2)
    plt.twinx()
    color = 'purple'
    plt.ylabel(r'Transfer Entropy $h^{C\rightarrow A}$', color=color)
    plt.errorbar(mulist,ratea,xerr=None,yerr=vara,color=color,fmt='o',elinewidth=2.5,capsize=7,markersize=9)
    plt.fill_between(mulist,ratea-vara,ratea+vara,color=color,alpha=0.1)
    plt.tick_params(axis='y', labelcolor=color)
    plt.vlines(0.35,0,0.06,color='grey',linestyle='dashed',alpha=0.8,linewidth=3.5)
    plt.axis([0,1.5,0,0.01])
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.tight_layout()
    plt.savefig('TransferEntropies.pdf',dpi=250)
    plt.show()
#%%
if __name__ == '__main__': #Creating data and plotting the histograms
    mu=0.01 #first 
    params=[K,eps,mu]
    iniconds=[1,1,1]
    const={'k1': c0*c0*c0/(K*K*K+c0*c0*c0)*mu+eps, 'k2': 0.1, 'k3': 1, 'k4': 0.1, 'k5': 0.1, 'k6': 0.1}
    const=list(const.values())
    # atdata,btdata,ctdata,b1acdata,b2acdata,a1cdata,a2cdata,b1cdata,b2cdata,abcdata,b1adata,b2adata,c1adata,c2adata,miadata,micdata,midata=parallelBS(core,MC,iniconds,const,params,laenge,timevec)
    plt.figure(1,figsize=(10,10))
    plt.hist(np.transpose(ctdata)[-10],bins=31,facecolor='b',alpha=0.75,density='True',align='left')
    plt.xlabel('Copy Number of C')
    plt.ylabel('Relative Frequency')
    plt.xlim(-0.5,30)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.grid(linewidth=2)
    plt.tight_layout()
    # plt.savefig('hist-mu=1e-3.pdf',dpi=250)
    plt.show()
#%%
if __name__ == '__main__':    
    mu=0.55 #second
    params=[K,eps,mu]
    iniconds=[1,10,10]
    const={'k1': c0*c0*c0/(K*K*K+c0*c0*c0)*mu+eps, 'k2': 0.1, 'k3': 1, 'k4': 0.1, 'k5': 0.1, 'k6': 0.1}
    const=list(const.values())
    # atdata,btdata,ctdata,b1acdata,b2acdata,a1cdata,a2cdata,b1cdata,b2cdata,abcdata,b1adata,b2adata,c1adata,c2adata,miadata,micdata,midata=parallelBS(core,MC,iniconds,const,params,laenge,timevec)
    plt.figure(1,figsize=(10,10))
    plt.hist(np.transpose(ctdata)[-10],bins=31,facecolor='b',alpha=0.75,density='True',align='left')
    plt.xlabel('Copy Number of C')
    plt.ylabel('Relative Frequency')
    plt.xlim(-0.5,130)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.grid(linewidth=2)
    plt.tight_layout()
    # plt.savefig('hist-mu=55e-2.pdf',dpi=250)
    plt.show()
 #%%  
if __name__ == '__main__': 
    mu=1 #third
    params=[K,eps,mu]
    iniconds=[10,100,100]
    const={'k1': c0*c0*c0/(K*K*K+c0*c0*c0)*mu+eps, 'k2': 0.1, 'k3': 1, 'k4': 0.1, 'k5': 0.1, 'k6': 0.1}
    const=list(const.values())
    # atdata,btdata,ctdata,b1acdata,b2acdata,a1cdata,a2cdata,b1cdata,b2cdata,abcdata,b1adata,b2adata,c1adata,c2adata,miadata,micdata,midata=parallelBS(core,MC,iniconds,const,params,laenge,timevec)
    plt.figure(1,figsize=(10,10))
    plt.hist(np.transpose(ctdata)[-10],bins=31,facecolor='b',alpha=0.75,density='True',align='left')
    plt.xlabel('Copy Number of C')
    plt.ylabel('Relative Frequency')
    plt.xlim(-0.5,200)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.grid(linewidth=2)
    plt.tight_layout()
    # plt.savefig('hist-mu=1.pdf',dpi=250)
    plt.show()
#%%
if __name__ == '__main__':
    y0=[10,100,100]
    klist=np.linspace(0.00001,20,2000)
    zeit=np.linspace(0,200,200)
    gzeitmu1=np.zeros(len(klist))
    pzeitmu1=np.zeros(len(klist))
    for i in range(len(klist)):
        params=[K,eps,klist[i]] 
        gzeitmu1[i]=odeint(bistableswitch,y0,zeit,args=(const,params) )[-1,0]
        pzeitmu1[i]=odeint(bistableswitch,y0,zeit,args=(const,params) )[-1,2]
        
    y0=[0,0,0]
    gzeitmu=np.zeros(len(klist))
    pzeitmu=np.zeros(len(klist))
    for i in range(len(klist)):
        params=[K,eps,klist[i]]
        gzeitmu[i]=odeint(bistableswitch,y0,zeit,args=(const,params) )[-1,0]
        pzeitmu[i]=odeint(bistableswitch,y0,zeit,args=(const,params) )[-1,2]
#%%     
if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rc('font',size=30)
    plt.rcParams['ps.useafm']=True
    plt.rcParams['axes.linewidth'] = 2
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['FreeSans']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(1,figsize=(10,10))
    # plt.plot(stableklist,stablefixpoints,'go',markersize=7)
    # plt.plot(unstableklist,unstablefixpoints,'1r',markersize=10)
    plt.plot(klist,pzeitmu,'b',linewidth=2.5)
    plt.plot(klist,pzeitmu1,'g',linewidth=2.5)
    plt.xlabel(r'Parameter $\mu$')
    plt.ylabel(r'Expectation Value of $c$')
    # plt.legend(('Stable fixpoint', 'Unstable fixpoint','Mean field approx.'),loc='best')
    plt.legend((r'$a_0$=0', r'$a_0$=10'),loc='best')
    # plt.axis([0.3,0.4,0,50])
    plt.grid(linewidth=2)
    plt.tight_layout()
    # plt.savefig('Bifurkation.pdf',dpi=250)
    plt.show()
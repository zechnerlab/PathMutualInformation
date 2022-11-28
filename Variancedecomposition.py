#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:29:09 2022

@author: moor
"""


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

if __name__ == '__main__':
    N=MC
    M=MC
    timevec=np.linspace(0,1000,300) #integration time
    mulist=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5] #list containing the values of mu to which we determine the rate
    vardiff_kc=[] #Var[k(c)]-Var[E[k(c)|A]]
    expvar_kc=[] #E[Var[k(c)|A]]
    vardiff_b=[]
    expvar_b=[]
    expvar_bac=[] #E[Var[B|AC]]
    expvar_bc=[] #E[Var[B|C]]
    expvar_ca=[]
    exp_var_kca_var=[] #error of E[Var[k(c)|A]]
    exp_var_bac_var=[] #error of E[Var[B|AC]]
    exp_var_bc_var=[] #error of E[Var[B|C]]
    exp_var_ca_var=[]
    var_bac_fin=[] #Var[B|AC]
    var_bc_fin=[] #Var[B|C]
    var_b_fin=[] #Var[B]
    var_c_fin=[]
    var_ca_fin=[]
    var_kca_fin=[]
    
    varof_varbac=[] #standard deviation of the variances
    varof_varbc=[]
    varof_varkc=[]
    varof_varca=[]
    
    for j in range(len(mulist)):
        mu=mulist[j]
        params=[K,eps,mu]
        atdata,btdata,ctdata,b1acdata,b2acdata,a1cdata,a2cdata,b1cdata,b2cdata,abcdata,b1adata,b2adata,c1adata,c2adata,miadata,micdata,midata=parallelisationBS(core,MC,iniconds,const,params,int(len(timevec)+1),timevec)  
        var_bac=MonteCarlo(b1acdata,MC)[1]
        var_bc=MonteCarlo(b1cdata,MC)[1]
        var_b=MonteCarlo(btdata,MC)[1]
        var_c=MonteCarlo(ctdata,MC)[1]
        var_ca=MonteCarlo(c1adata,MC)[1]
        exp_var_bac_data=[]
        exp_var_bc_data=[]
        exp_var_ca_data=[]
        exp_var_bac=np.zeros(len(b2acdata[0]))
        exp_var_bc=np.zeros(len(b2cdata[0]))
        exp_var_ca=np.zeros(len(b2cdata[0]))
        exp_var_ca_data=[]
        for i in range(len(c1adata)):
            exp_var_bac+=(b2acdata[i]-b1acdata[i]*b1acdata[i])/MC
            exp_var_bc+=(b2cdata[i]-b1cdata[i]*b1cdata[i])/MC
            exp_var_bac_data.append(b2acdata[i]-b1acdata[i]*b1acdata[i])
            exp_var_bc_data.append(b2cdata[i]-b1cdata[i]*b1cdata[i])
            exp_var_ca+=(c2adata[i]-c1adata[i]*c1adata[i])/MC
            exp_var_ca_data.append(c2adata[i]-c1adata[i]*c1adata[i])
        vardiff_b.append(var_bac[-2]-var_bc[-2])
        var_bac_fin.append(var_bac[-2])
        var_bc_fin.append(var_bc[-2])
        var_b_fin.append(var_b[-2])
        var_c_fin.append(var_c[-2])
        var_ca_fin.append(var_ca[-2])
        expvar_b.append(exp_var_bc[-2]-exp_var_bac[-2])
        expvar_bac.append(exp_var_bac[-2])
        expvar_bc.append(exp_var_bc[-2])
        expvar_ca.append(exp_var_ca[-2])
        exp_var_bac_var.append(MonteCarlo(exp_var_bac_data,MC)[1][-2])
        exp_var_bc_var.append(MonteCarlo(exp_var_bc_data,MC)[1][-2])
        exp_var_ca_var.append(MonteCarlo(exp_var_ca_data,MC)[1][-2])
        
        vardiff_b_temp=np.zeros(len(ctdata[0])) #calculate standard deviation of the variances
        vardiffbac_temp=np.zeros(len(ctdata[0]))
        vardiffbc_temp=np.zeros(len(ctdata[0]))
        vardiffca_temp=np.zeros(len(ctdata[0]))
        var_bac_tempdata=np.zeros(M)
        var_bc_tempdata=np.zeros(M)
        var_b_tempdata=np.zeros(M)
        var_c_tempdata=np.zeros(M)
        var_ca_tempdata=np.zeros(M)
        for i in range(M):
            var_bac_temp=np.zeros(len(ctdata[0]))
            av_bac_temp=np.zeros(len(ctdata[0]))
            var_bc_temp=np.zeros(len(ctdata[0]))
            av_bc_temp=np.zeros(len(ctdata[0]))
            var_b_temp=np.zeros(len(ctdata[0]))
            av_b_temp=np.zeros(len(ctdata[0]))
            var_c_temp=np.zeros(len(ctdata[0]))
            av_c_temp=np.zeros(len(ctdata[0]))
            var_ca_temp=np.zeros(len(ctdata[0]))
            av_ca_temp=np.zeros(len(ctdata[0]))
            for j in range(N):
                n=int(rnd.uniform(0.0,MC))
                av_bac_temp+=b1acdata[n]/N
                var_bac_temp+=b1acdata[n]*b1acdata[n]/N
                av_bc_temp+=b1cdata[n]/N
                var_bc_temp+=b1cdata[n]*b1cdata[n]/N
                av_b_temp+=btdata[n]/N
                var_b_temp+=btdata[n]*btdata[n]/N
                av_c_temp+=ctdata[n]/N
                var_c_temp+=ctdata[n]*ctdata[n]/N
                av_ca_temp+=c1adata[n]/N
                var_ca_temp+=c1adata[n]*c1adata[n]/N
            var_bac_tempdata[i]=(var_bac_temp[-2]-av_bac_temp[-2]*av_bac_temp[-2])
            var_bc_tempdata[i]=(var_bc_temp[-2]-av_bc_temp[-2]*av_bc_temp[-2])
            var_b_tempdata[i]=(var_b_temp[-2]-av_b_temp[-2]*av_b_temp[-2])
            var_c_tempdata[i]=(var_c_temp[-2]-av_c_temp[-2]*av_c_temp[-2])
            var_ca_tempdata[i]=(var_ca_temp[-2]-av_ca_temp[-2]*av_ca_temp[-2])
        varof_varbac.append(np.std(var_b_tempdata-var_bac_tempdata))
        varof_varbc.append(np.std(var_b_tempdata-var_bc_tempdata))
        varof_varca.append(np.std(var_c_tempdata-var_ca_tempdata))
        
    exp_var_bdiff_var=[]
    for i in range(len(exp_var_bac_var)):
        exp_var_bdiff_var.append(exp_var_bc_var[i]-exp_var_bac_var[i])

    vardiff_bac=np.zeros(len(var_b_fin))
    vardiff_bc=np.zeros(len(var_b_fin))
    vardiff_ca=np.zeros(len(var_c_fin))
    for i in range(len(var_b_fin)):
        vardiff_bac[i]=var_b_fin[i]-var_bac_fin[i]
        vardiff_bc[i]=var_b_fin[i]-var_bc_fin[i]
        vardiff_ca[i]=var_c_fin[i]-var_ca_fin[i]
        
#Plotting
if __name__ == '__main__':
    def plot(x,y,xlabl,ylabl,var=False,legend=False,colors=False,axis=False,errorbar=False,errorvar=False,errorcolors=False):
        lwd=3
        matplotlib.rc('font',size=30)
        plt.rcParams['ps.useafm']=True
        matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['FreeSans']})
        plt.rcParams['pdf.fonttype'] = 42
        plt.figure(1,figsize=(10,10))
        if y:
            if colors:
                for i in range(len(y)):
                    plt.plot(x,y[i],colors[i],linewidth=lwd,markersize=9)
                    if var:
                        plt.fill_between(x,y[i]-np.sqrt(var[i]-np.array(y[i])*np.array(y[i])),y[i]+np.sqrt(var[i]-np.array(y[i])*np.array(y[i])),color=colors[i],alpha=0.3)
            else:
                for i in range(len(y)):
                    plt.plot(x,y[i],linewidth=lwd)
                    if var:
                        plt.fill_between(x,y[i]-np.sqrt(var[i]-np.array(y[i])*np.array(y[i])),y[i]+np.sqrt(var[i]-np.array(y[i])*np.array(y[i])),color=colors[i],alpha=0.3)
        if errorbar:
            if errorcolors:
                for i in range(len(errorbar)):
                    if errorvar:
                        plt.errorbar(x,errorbar[i],xerr=None,yerr=errorvar[i],color=errorcolors[i],fmt='o',elinewidth=2.5,capsize=7,markersize=9)
                    else:
                        plt.errorbar(x,errorbar[i],xerr=None,yerr=None,color=errorcolors[i],fmt='o',elinewidth=2.5,capsize=7,markersize=9)
            else: 
                for i in range(len(errorbar)):
                    if errorvar:
                        plt.errorbar(x,errorbar[i],xerr=None,yerr=errorvar[i],fmt='o',elinewidth=2.5,capsize=7,markersize=9)
                    else:
                        plt.errorbar(x,errorbar[i],xerr=None,yerr=None,fmt='o',elinewidth=2.5,capsize=7,markersize=9)
                
        if legend:
            plt.legend(legend, loc='best')
        if axis:
            plt.axis(axis)
        plt.xlabel(xlabl)
        plt.ylabel(ylabl)
        plt.grid(linewidth=2)
        plt.tight_layout()
        return  
    
    plot(mulist,y=False,xlabl=r'Feedback Strength $\mu$',ylabl='Variance',legend=(r'Var[$C(t)$]-Var[E[$C(t)|A_0^t$]]',r'E[Var[$C(t)|A_0^t$]]'),errorbar=[vardiff_ca,expvar_ca],axis=[0,1.5,0,0.0022],errorvar=[np.array(varof_varca)*2.5,np.sqrt(exp_var_ca_var)/np.sqrt(MC)*2.5],errorcolors=['g','purple'])#np.sqrt(varof_vardiff_kc)/np.sqrt(K)*2.5
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) #varof_vardiff_kc
    plt.savefig('VarianceDecomposition_kc.png', dpi=250)
    plt.show()
    plot(mulist,y=False,xlabl=r'Feedback Strength $\mu$',ylabl='Variance',legend=('Var[$B(t)$]-Var[E[$B(t)|A_0^t,C_0^t$]]','E[Var[$B(t)|A_0^t,C_0^t$]]'),axis=[0,1.5,0,130],errorbar=[vardiff_bac,expvar_bac],errorvar=[np.array(varof_varbac)*2.5,np.sqrt(exp_var_bac_var)/np.sqrt(MC)*2.5],errorcolors=['g','purple'])
    plt.savefig('VarianceDecomposition_bac.png', dpi=250) #varof_varbac
    plt.show()
    plot(mulist,y=False,xlabl=r'Feedback Strength $\mu$',ylabl='Variance',legend=('Var[$B(t)$]-Var[E[$B(t)|C_0^t$]]','E[Var[$B(t)|C_0^t$]]'),axis=[0,1.5,0,300],errorbar=[vardiff_bc,expvar_bc],errorvar=[np.array(varof_varbc)*2.5,np.sqrt(exp_var_bc_var)/np.sqrt(MC)*2.5],errorcolors=['g','purple'])
    plt.savefig('VarianceDecomposition_bc.png', dpi=250) #varof_varbc
    plt.show()
    plot(mulist,y=False,xlabl=r'Feedback Strength $\mu$',ylabl='Variance',legend=(r'Var[$C(t)$]-Var[E[$C(t)|A_0^t$]]',r'E[Var[$C(t)|A_0^t$]]'),errorbar=[vardiff_ca,expvar_ca],axis=[0,1.5,0,260],errorvar=[np.array(varof_varca)*2.5,np.sqrt(exp_var_ca_var)/np.sqrt(MC)*2.5],errorcolors=['g','purple'])#np.sqrt(varof_vardiff_kc)/np.sqrt(K)*2.5
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
    plt.savefig('VarianceDecomposition_ca.png', dpi=250) #varof_varca
    plt.show()

# Distance between k(c) obtained via Gamma distribution and via Taylor expansion

if __name__ == '__main__':
    N_dist=500
    dist=[]
    dist_std=[]
    dist_10quant=[]
    dist_90quant=[]
    dist_median=[]
    dist_temp_data=[]
    for m in range(0,len(mulist)):
        mu=mulist[m]
        atdata,btdata,ctdata,b1acdata,b2acdata,a1cdata,a2cdata,b1cdata,b2cdata,abcdata,b1adata,b2adata,c1adata,c2adata,miadata,micdata,midata=parallelisationBS(core,MC,iniconds,const,params,int(len(timevec)+1),timevec)
        kca_gamma=np.zeros(N_dist)
        kca_taylor=np.zeros(N_dist)
        dist_temp=[]
        for k in range(N_dist):
            j=int(rnd.uniform(0.0,MC))
            kca_temp=0
            for i in range(MC):
                n=np.random.gamma(np.abs(c1adata[j]*c1adata[j]/(c2adata[j]-c1adata[j]*c1adata[j])),(c2adata[j]-c1adata[j]*c1adata[j])/c1adata[j])
                kca_temp+=(mu*n*n*n/(K*K*K+n*n*n)+eps)/MC
            kca_gamma[k]=(kca_temp[-2])
            kca_taylor_temp=mu*c1adata[j]*c1adata[j]*c1adata[j]/(K*K*K+c1adata[j]*c1adata[j]*c1adata[j])+eps
            kca_taylor[k]=(kca_taylor_temp[-2])
            dist_temp.append(np.abs(kca_taylor[k]-kca_gamma[k])/kca_gamma[k])
        dist_temp_data.append(dist_temp)
        dist.append(np.mean(dist_temp))
        dist_median.append(np.quantile(dist_temp,0.5))
        dist_10quant.append(np.quantile(dist_temp,0.1))
        dist_90quant.append(np.quantile(dist_temp,0.9))
        dist_std.append(np.std(dist_temp))

if __name__ == '__main__':
    dist_quant=np.zeros((2,len(dist_10quant)))
    dist_quant[0]=np.array(dist_median)-np.array(dist_10quant)
    dist_quant[1]=np.array(dist_90quant)-np.array(dist_median)

if __name__ == '__main__':
    plot(mulist,y=False,xlabl=r'Feedback Strength $\mu$',ylabl=r'Distance $d$',errorbar=[dist_median],axis=[0,1.5,0,0.2],errorvar=[dist_quant],errorcolors=['b'])
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.savefig('dist_quantil.png',dpi=250)
    plt.show()




    

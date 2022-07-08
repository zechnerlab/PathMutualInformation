#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:23:58 2022

@author: moor
"""

import numpy as np
from functools import partial
import random as rnd
from scipy.integrate import odeint
from multiprocessing import Pool

def evolveBS(y,t,g,p,constants,K,eps,mu): 
    '''
    set of differential equations for the continous parts of the moment equations and path mutual information of the bistable switch system.
    '''
    r1gp,r2gp,g1p,g2p,r1p,r2p,grp,r1g,r2g,p1g,p2g,rpg,mig,mip=y
    c_1,c_2,c_3,c_4,c_5,c_6=constants
    c_1p=p*p*p/float(K*K*K+p*p*p)*mu+eps
    dydt=[c_3*g-c_4*r1gp-c_5*r2gp+c_5*r1gp*r1gp, #r1gp 
          2*c_3*g*r1gp+c_3*g-2*c_4*r2gp+c_4*r1gp-2*c_5*r2gp/r1gp*(r2gp-r1gp*r1gp), #r2gp
          c_1p-c_2*g1p-c_5*grp+c_5*g1p*r1p, #g1p
          2*c_1p*g1p+c_1p-2*c_2*g2p+c_2*g1p-2*c_5*g2p*grp/g1p+c_5*g2p*r1p*2, #g2p
          c_3*g1p-c_4*r1p-c_5*r2p+c_5*r1p*r1p, #r1p
          2*c_3*grp+c_3*g1p-c_4*2*r2p+c_4*r1p-2*c_5*r2p*r2p/r1p+c_5*r1p*r2p*2, #r2p
          c_1p*r1p-c_2*grp+c_3*g2p-c_4*grp-2*c_5*r2p*grp/r1p+c_5*r2p*g1p+c_5*r1p*grp, #grp 
          c_3*g-c_4*r1g-(3*mu*p1g*p1g/(K*K*K+p1g*p1g*p1g)-3*mu*p1g*p1g*p1g*p1g*p1g/((K*K*K+p1g*p1g*p1g)*(K*K*K+p1g*p1g*p1g)))*(rpg-p1g*r1g), #r1g
          2*c_3*g*r1g+c_3*g-2*c_4*r2g+c_4*r1g-(3*mu*p1g*p1g/(K*K*K+p1g*p1g*p1g)-3*mu*p1g*p1g*p1g*p1g*p1g/((K*K*K+p1g*p1g*p1g)*(K*K*K+p1g*p1g*p1g)))*((2*r2g*rpg)/r1g-2*p1g*r2g), #r2g
          c_5*r1g-c_6*p1g-(3*mu*p1g*p1g/(K*K*K+p1g*p1g*p1g)-3*mu*p1g*p1g*p1g*p1g*p1g/((K*K*K+p1g*p1g*p1g)*(K*K*K+p1g*p1g*p1g)))*(p2g-p1g*p1g), #p1g
          2*c_5*rpg+c_5*r1g-2*c_6*p2g+c_6*p1g-(3*mu*p1g*p1g/(K*K*K+p1g*p1g*p1g)-3*mu*p1g*p1g*p1g*p1g*p1g/((K*K*K+p1g*p1g*p1g)*(K*K*K+p1g*p1g*p1g)))*(2*p2g*p2g/p1g-2*p1g*p2g), #p2g
          c_3*g*p1g-c_4*rpg+c_5*r2g-c_6*rpg-(3*mu*p1g*p1g/(K*K*K+p1g*p1g*p1g)-3*mu*p1g*p1g*p1g*p1g*p1g/((K*K*K+p1g*p1g*p1g)*(K*K*K+p1g*p1g*p1g)))*(2*p2g*rpg/p1g-p2g*r1g-p1g*rpg), #rpg
          -(c_1p-(mu*p1g*p1g*p1g/(K*K*K+p1g*p1g*p1g)+eps)), #mig
          -c_5*(r1gp-r1p) #mip
          ] 
    return dydt

def computeTrajectoryBS(n,iniconds,const,params,laenge,destime):
    '''
    Integrates path mutual information and the required conditional moments of B given A and C and of A, B given A and B given C over time 
    and determines the SSA-trajectory of the species

    Parameters
    ----------
    n : integer; for multiprocessing
    iniconds : list with three integer-entries; initial conditions of our species
    const : list; reaction constants of the system 
    params : list with three entries containing the values of K, epsilon and mu
    laenge : integer, length of the trajectories
    destime : array of float; time vector of the integration

    Returns 
    -------
    Array of single trajectory.
    trajectory of A, of B, of C, trajectory the expected value of of A conditional on B and C, its second moment, expected value of A given C, its second moment, B given C, its second moment, C given A, its second moment, B given A, its second moment, transfer entropy C->A, transfer entropy A->C, path mutual information
        
    '''
    g0,r0,p0=iniconds #assigning initial conditions 
    K,eps,mu=params
    gt,rt,pt,r1gp,r2gp,g1p,g2p,r1p,r2p,grp,r1g,r2g,p1g,p2g,rpg,mig,mip,mi=[np.zeros(laenge) for i in range(18)] #definition of the trajectories
    updates=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]] #stoichiometric changes

    t=0 #determines the whole time that has already passed 
    tau=0 #determines the actual time point for the numerical integration
    g=g0
    r=r0
    p=p0
    gt[0]=g0
    rt[0]=r0
    pt[0]=p0
    t_max=destime[-1]
    j=0
    h=0
    r1gp[0]=r0
    r2gp[0]=r0*r0
    g1p[0]=g0
    g2p[0]=g0*g0
    r1p[0]=r0
    r2p[0]=r0*r0
    grp[0]=r0*g0
    sol=np.array([[r0,r0*r0,g0,g0*g0,r0,r0*r0,g0*r0,r0,r0*r0,p0,p0*p0,r0*p0,0,0]])
    while (t<t_max): 
        
        const[0]=p*p*p/float(K*K*K+p*p*p)*mu+eps  
        propensities=[const[0], const[1]*g, const[2]*g, const[3]*r, const[4]*r, const[5]*p] #SSA 
        reacsum=sum(propensities) #SSA 
        time=-np.log(rnd.uniform(0.0,1.0))/float(reacsum) #SSA 
        RV=rnd.uniform(0.0,1.0)*reacsum     #SSA  
        index=1 #SSA 
        value=propensities[0] #SSA 
        while value<RV: #SSA 
            index+=1 #SSA  
            value+=propensities[index-1]  #SSA  
            
        if h<len(destime): #step-wise integration algorithm of the conditional probability distributions and the path mutual information 
            while destime[h]<(t+time):  
                y0=[sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4],sol[-1,5],sol[-1,6],sol[-1,7],sol[-1,8],sol[-1,9],sol[-1,10],sol[-1,11],sol[-1,12],sol[-1,13]]
                timevec=np.linspace(tau,destime[h],5) #integration interval
                sol=odeint(evolveBS,y0,timevec, args=(g,p,const,K,eps,mu),rtol=1e-6) #integration from the actual time point until the next desired time point
                tau=destime[h] #update the actual time point from which the next integration starts
                r1gp[h]=sol[-1,0] #update of the species
                r2gp[h]=sol[-1,1]
                g1p[h]=sol[-1,2]
                g2p[h]=sol[-1,3]
                r1p[h]=sol[-1,4]
                r2p[h]=sol[-1,5]
                grp[h]=sol[-1,6]
                r1g[h]=sol[-1,7]
                r2g[h]=sol[-1,8]
                p1g[h]=sol[-1,9]
                p2g[h]=sol[-1,10]
                rpg[h]=sol[-1,11]
                mig[h]=sol[-1,12]
                mip[h]=sol[-1,13]
                mi[h]=mig[h]+mip[h]
                gt[h]=g
                rt[h]=r
                pt[h]=p
                h=h+1
                j=j+1
                if h>=len(destime)-1:
                    break
            y0=[sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4],sol[-1,5],sol[-1,6],sol[-1,7],sol[-1,8],sol[-1,9],sol[-1,10],sol[-1,11],sol[-1,12],sol[-1,13]]

            timevec=np.linspace(tau,t+time,5)
            sol=odeint(evolveBS,y0,timevec, args=(g,p,const,K,eps,mu),rtol=1e-6)  #integration from the actual time point until the next desired time point 
            tau=t+time #update the actual time point from which the next integration starts
        t=t+time #update of the total time 
        if index==5: #evaluation of the stochastic integral for the jumps (reaction 5)
            r1gp[h]=sol[-1,0]+(sol[-1,1]-sol[-1,0]*sol[-1,0])/sol[-1,0]
            r2gp[h]=sol[-1,1]+(2*sol[-1,1]*sol[-1,1]/sol[-1,0]-sol[-1,0]*sol[-1,1]*2)/sol[-1,0]
            g1p[h]=sol[-1,2]+(sol[-1,6]-sol[-1,2]*sol[-1,4])/sol[-1,4]
            g2p[h]=sol[-1,3]+(2*(sol[-1,3]*sol[-1,6]/sol[-1,2])-sol[-1,3]*sol[-1,4]*2)/sol[-1,4]
            r1p[h]=sol[-1,4]+(sol[-1,5]-sol[-1,4]*sol[-1,4])/sol[-1,4]
            r2p[h]=sol[-1,5]+(2*sol[-1,5]*sol[-1,5]/sol[-1,4]-2*sol[-1,5]*sol[-1,4])/sol[-1,4]
            grp[h]=sol[-1,6]+(2*sol[-1,5]*sol[-1,6]/sol[-1,4]-sol[-1,5]*sol[-1,2]-sol[-1,4]*sol[-1,6])/sol[-1,4]
            
            if sol[-1,0]==0 and sol[-1,4]!=0:
                mip[h]=sol[-1,13]-np.log(const[4]*sol[-1,4])
            if sol[-1,0]!=0 and sol[-1,4]==0:   
                mip[h]=sol[-1,13]+np.log(const[4]*sol[-1,0])
            if sol[-1,0]==0 and sol[-1,4]==0:
                mip[h]=sol[-1,13]
            if sol[-1,0]!=0 and sol[-1,4]!=0:
                mip[h]=sol[-1,13]+np.log(const[4]*sol[-1,0])-np.log(const[4]*sol[-1,4])
            sol[-1,0]=r1gp[h] #r1gp,r2gp,g1p,r1p,r2p,grp,r1g,r2g,p1g,p2g,rpg,mig,mip
            sol[-1,1]=r2gp[h]
            sol[-1,2]=g1p[h]
            sol[-1,3]=g2p[h]
            sol[-1,4]=r1p[h]
            sol[-1,5]=r2p[h]
            sol[-1,6]=grp[h] 
            sol[-1,13]=mip[h]

                
        if index==1: #evaluation of the stochastic integral for the jumps (reaction 1)
            lambdagp=(mu*p*p*p/float(K*K*K+p*p*p)+eps)
            lambdag=(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
            r1g[h]=sol[-1,7]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(sol[-1,11]-sol[-1,9]*sol[-1,7])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
            r2g[h]=sol[-1,8]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(2*sol[-1,8]*sol[-1,11]/sol[-1,7]-2*sol[-1,8]*sol[-1,9])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
            p1g[h]=sol[-1,9]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(sol[-1,10]-sol[-1,9]*sol[-1,9])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
            p2g[h]=sol[-1,10]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(2*sol[-1,10]*sol[-1,10]/sol[-1,9]-2*sol[-1,10]*sol[-1,9])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
            rpg[h]=sol[-1,11]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(2*sol[-1,10]*sol[-1,11]/sol[-1,9]-sol[-1,10]*sol[-1,7]-sol[-1,9]*sol[-1,11])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
            if lambdagp==0 and lambdag!=0:
                mig[h]=sol[-1,12]-np.log(lambdag)
            if lambdagp!=0 and lambdag==0:
                mig[h]=sol[-1,12]+np.log(lambdagp)
            if lambdagp==0 and lambdag==0:
                mig[h]=sol[-1,12]
            if lambdagp!=0 and lambdag!=0:
                mig[h]=sol[-1,12]+np.log(lambdagp)-np.log(lambdag)
            sol[-1,7]=r1g[h]
            sol[-1,8]=r2g[h]
            sol[-1,9]=p1g[h]
            sol[-1,10]=p2g[h]
            sol[-1,11]=rpg[h]
            sol[-1,12]=mig[h]
            
        mi[h]=mig[h]+mip[h]
        g+=updates[index-1][0] #update of the SSA trajectories
        r+=updates[index-1][1]
        p+=updates[index-1][2]
    return gt,rt,pt,r1gp,r2gp,g1p,g2p,r1p,r2p,grp,r1g,r2g,p1g,p2g,rpg,mig,mip,mi

def parallelisationBS(core,MC,iniconds,const,params,laenge,timevec):
    '''
    Parallelisation of the Monte Carlo integration of the path mutual information between A and C

    Parameters
    ----------
    core : integer; number of cores of the computer that are used for the calculation
    MC : integer; sample size of the Monte Carlo integration
    iniconds : list with three integer-entries; initial conditions of our species
    const : list; reaction constants of the system
    params : list with three entries containing the values of K, epsilon and mu
    laenge : integer, length of the trajectories
    timevec : array of float; time vector of the integration
    
    Returns 
    -------
    The output contains an arrays of all the generated single trajectories.
    trajectory of A, of B, of C, trajectory the expected value of of A conditional on B and C, its second moment, expected value of A given C, its second moment, B given C, its second moment, C given A, its second moment, B given A, its second moment, transfer entropy C->A, transfer entropy A->C, path mutual information
    
    '''
    gtdata=[] #definition of the final arrays
    rtdata=[]
    ptdata=[]
    r1gpdata=[]
    r2gpdata=[]
    r1pdata=[]
    r2pdata=[]
    g1pdata=[]
    g2pdata=[]
    grpdata=[]
    r1gdata=[]
    r2gdata=[]
    p1gdata=[]
    p2gdata=[]
    migdata=[]
    mipdata=[]
    midata=[]
    with Pool(core) as pool: #parallelisation
        results = pool.map(partial(computeTrajectoryBS,iniconds=iniconds,const=const,params=params,laenge=laenge,destime=timevec), [k for k in range(MC)])
        for r in results:
            gt_temp,rt_temp,pt_temp,r1gp_temp,r2gp_temp,g1p_temp,g2p_temp,r1p_temp,r2p_temp,grp_temp,r1g_temp,r2g_temp,p1g_temp,p2g_temp,rpg_temp,mig_temp,mip_temp,mi_temp = r
            gtdata.append(gt_temp[:-1]) #collecting all the single trajectories 
            rtdata.append(rt_temp[:-1])
            ptdata.append(pt_temp[:-1])
            r1gpdata.append(r1gp_temp[:-1])
            r2gpdata.append(r2gp_temp[:-1])
            r1pdata.append(r1p_temp[:-1])
            r1gdata.append(r1g_temp[:-1])
            r2gdata.append(r2g_temp[:-1])
            r2pdata.append(r2p_temp[:-1])
            g1pdata.append(g1p_temp[:-1])
            g2pdata.append(g2p_temp[:-1])
            grpdata.append(grp_temp[:-1])
            p1gdata.append(p1g_temp[:-1])
            p2gdata.append(p2g_temp[:-1])
            migdata.append(mig_temp[:-1])
            mipdata.append(mip_temp[:-1])
            midata.append(mi_temp[:-1])
        return gtdata,rtdata,ptdata,r1gpdata,r2gpdata,g1pdata,g2pdata,r1pdata,r2pdata,grpdata,r1gdata,r2gdata,p1gdata,p2gdata,migdata,mipdata,midata
    
def MonteCarlo(data,MC):
    '''
    generates a Monte Carlo average of the input data
    '''
    average=np.zeros(len(data[0]))
    variance=np.zeros(len(data[0]))
    for i in range(MC):
        average+=data[i]
        variance+=data[i]*data[i]
    average=average/MC
    variance=variance/MC
    variance=np.sqrt(variance-average*average)
    return average,variance
        
 
def bistableswitch(y,t,c,params):
    '''
    mean field equations of the bistable switch system needed to obtain the bifurcation plot
    '''
    g,r,p=y
    K,eps,mu=params
    dydt=[eps+mu*p*p*p/float(K*K*K+p*p*p)-c[1]*g,
          g*c[2]-r*c[3],
          r*c[4]-p*c[5]]
    return dydt
    
    
    
    
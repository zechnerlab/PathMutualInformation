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

def BS(y,t,g,p,constants,K,eps,mu):
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

def trajectoryBS(n,iniconds,const,params,laenge,destime):
    g0,r0,p0=iniconds
    K,eps,mu=params
    gt,rt,pt,r1gp,r2gp,g1p,g2p,r1p,r2p,grp,r1g,r2g,p1g,p2g,rpg,mig,mip,mi=[np.zeros(laenge) for i in range(18)]
    updates=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]

    t=0
    tau=0
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
        propensities=[const[0], const[1]*g, const[2]*g, const[3]*r, const[4]*r, const[5]*p] 
        reacsum=sum(propensities)
        time=-np.log(rnd.uniform(0.0,1.0))/float(reacsum)
        RV=rnd.uniform(0.0,1.0)*reacsum      
        index=1
        value=propensities[0]
        while value<RV:
            index+=1
            value+=propensities[index-1]  
        if h<len(destime):
            while destime[h]<(t+time):  
                y0=[sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4],sol[-1,5],sol[-1,6],sol[-1,7],sol[-1,8],sol[-1,9],sol[-1,10],sol[-1,11],sol[-1,12],sol[-1,13]]
                timevec=np.linspace(tau,destime[h],5)
                sol=odeint(BS,y0,timevec, args=(g,p,const,K,eps,mu),rtol=1e-6)
                tau=destime[h]
                r1gp[h]=sol[-1,0]
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
            sol=odeint(BS,y0,timevec, args=(g,p,const,K,eps,mu),rtol=1e-6)
            tau=t+time
        t=t+time
        if index==5:
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

                
        if index==1:
            lambdagp=(mu*p*p*p/float(K*K*K+p*p*p)+eps)
            lambdag=(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
            r1g[h]=sol[-1,7]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(sol[-1,11]-sol[-1,9]*sol[-1,7])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
            r2g[h]=sol[-1,8]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(2*sol[-1,8]*sol[-1,11]/sol[-1,7]-2*sol[-1,8]*sol[-1,9])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
            p1g[h]=sol[-1,9]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(sol[-1,10]-sol[-1,9]*sol[-1,9])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
            p2g[h]=sol[-1,10]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(2*sol[-1,10]*sol[-1,10]/sol[-1,9]-2*sol[-1,10]*sol[-1,9])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
            rpg[h]=sol[-1,11]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(2*sol[-1,10]*sol[-1,11]/sol[-1,9]-sol[-1,10]*sol[-1,7]-sol[-1,9]*sol[-1,11])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
            #Ableitung (3*mu*p1g*p1g/(K*K*K+p1g*p1g*p1g)-3*mu*p1g*p1g*p1g*p1g*p1g/((K*K*K+p1g*p1g*p1g)*(K*K*K+p1g*p1g*p1g)))*(p-p1g)
            #Funktion mu*p1g*p1g*p1g/(K*K*K+p1g*p1g*p1g)+offset
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
        g+=updates[index-1][0]
        r+=updates[index-1][1]
        p+=updates[index-1][2]
    return gt,rt,pt,r1gp,r2gp,g1p,g2p,r1p,r2p,grp,r1g,r2g,p1g,p2g,rpg,mig,mip,mi

def parallelBS(core,MC,iniconds,const,params,laenge,timevec):
    gtdata=[]
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
    with Pool(core) as pool:
        results = pool.map(partial(trajectoryBS,iniconds=iniconds,const=const,params=params,laenge=laenge,destime=timevec), [k for k in range(MC)])
        for r in results:
            gt_temp,rt_temp,pt_temp,r1gp_temp,r2gp_temp,g1p_temp,g2p_temp,r1p_temp,r2p_temp,grp_temp,r1g_temp,r2g_temp,p1g_temp,p2g_temp,rpg_temp,mig_temp,mip_temp,mi_temp = r
            gtdata.append(gt_temp[:-1])
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
     g,r,p=y
     K,eps,mu=params
     dydt=[eps+mu*p*p*p/float(K*K*K+p*p*p)-c[1]*g,
           g*c[2]-r*c[3],
           r*c[4]-p*c[5]]
     return dydt
    
    
    
    
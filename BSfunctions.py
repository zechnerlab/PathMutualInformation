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
from functions import updateMatrixAC, updateMatrixC, dN, dN2D, mean2d
from exactsolutions import evolveQuasiExactBS

#The letters g,r,p denote the species A,B,C

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

def updateMatrixA(d,a,const,mA,params):
    '''
    integration matrix for the evaluation of the probability distribution conditional on C
    '''
    for b in range(d*d):
        for c in range(d*d):
            if c==(b+d): 
                mA[b][c]=(int(b/d)+1)*const[3]
            if b==(c+d):
                mA[b][c]=const[2]*a
            if c==(b+1) and (c % d)!=0:
                mA[b][c]=(c % d)*const[5]
            if c==(b-1) and (b % d)!=0:
                mA[b][c]=int(b/d)*const[4]
            if b==c:
                mA[b][c]=-(const[2]*a+const[3]*int(b/d)+const[4]*int(b/d)+const[5]*(c % d)+params[2]*(c % d)*(c % d)*(c % d)/(params[0]*params[0]*params[0]+(c % d)*(c % d)*(c % d))+params[1])
    return

def dN_hill(P,d,params):
    '''
    stochastic jump for two dimensional array
    '''
    jump=np.zeros(d)
    for i in range(1,d):
        jump[i]=params[2]*i*i*i/(params[0]*params[0]*params[0]+i*i*i)+params[1]-1
    if np.shape(P)!= (d,d):
        P=np.reshape(P,(d,d))
    for a in range(d):
        for b in range(d):
            P[a][b]=P[a][b]*jump[b]
    P=np.matrix.flatten(P)
    return P

def mean_hill(P,d,params):
    '''
    mean of the copy numbers stored in a two dimensional array
    '''
    if np.shape(P)!= (d,d):
        P=np.reshape(P,(d,d))
    mean=0
    for b in range(d):
        for a in range(d):
            mean+=(params[2]*b*b*b/(params[0]*params[0]*params[0]+b*b*b)+params[1])*P[a][b]
    return mean

def computeTrajectoryBS(n,iniconds,const,params,laenge,destime,exact=False,dim=50):
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
    a0,b0,c0=iniconds #assigning initial conditions 
    K,eps,mu=params
    if exact==True:
        abvec=np.zeros(dim*dim) #defines the lattice of the copy numbers of species A and B
        for i in range(dim*dim):
            abvec[i]=i%dim
        bcvec=np.zeros(dim*dim)
        for i in range(dim*dim):
            bcvec[i]=mu*i*i*i/(K*K*K+i*i*i)+eps
            
        Pt_mi=np.zeros((1,dim*dim+dim+1+dim*dim)) #array that will contain the conditional probability distributions and the mutual information, dimension will change
        dydt=np.zeros(dim*dim+dim+1+dim*dim) #for the numerical integration
        Pt_mi[0][b0]=1
        Pt_mi[0][dim+1+dim*a0+b0]=1
        Pt_mi[0][dim+1+dim*dim+dim*b0+c0]=1 
        at,bt,ct,mi_exact,b1ac,b2ac,a1c,a2c,b1c,b2c,abc,b1a,b2a,c1a,c2a,bca,mia,mic,mi=[np.zeros(laenge) for i in range(19)]
        updates=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]] #stoichiometric changes
        dPAC1,dPAC2=[np.zeros(dim) for i in range(2)] #for the numerical integration
        dPC1,dPC2,dPA1,dPA2=[np.zeros(dim*dim) for i in range(4)] #for the numerical integration
        
    
        t=0 #determines the whole time that has already passed 
        tau=0 #determines the actual time point for the numerical integration
        Pt_ac=np.zeros((laenge,dim)) #array that will contain all the probability distributions conditional on A and C of the different time points
        Pt_c=np.zeros((laenge,dim*dim)) #array that will contain all the probability distributions conditional on C of the different time points
        Pt_a=np.zeros((laenge,dim*dim)) #array that will contain all the probability distributions conditional on A of the different time points
        a=a0
        b=b0
        c=c0
        at[0]=a0
        bt[0]=b0
        ct[0]=c0
        t_max=destime[-1]
        mAC=np.zeros((dim,dim)) #integration matrix for Pt_ac
        updateMatrixAC(dim,a,const,mAC)
        mC=np.zeros((dim*dim,dim*dim)) #integration matrix for Pt_c
        updateMatrixC(dim,const,mC)
        mA=np.zeros((dim*dim,dim*dim)) #integration for Pt_a
        updateMatrixA(dim,a,const,mA,params)
        h=0
        sol=np.array([[b0,b0*b0,a0,a0*a0,b0,b0*b0,a0*b0,b0,b0*b0,c0,c0*c0,b0*c0,0,0]])
        while (t<t_max): 
            
            const[0]=c*c*c/float(K*K*K+c*c*c)*mu+eps  
            propensities=[const[0], const[1]*a, const[2]*a, const[3]*b, const[4]*b, const[5]*c] #SSA 
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
                    P0=Pt_mi[-1].copy()
                    timevec=np.linspace(tau,destime[h],5) #integration interval
                    sol=odeint(evolveBS,y0,timevec, args=(a,c,const,K,eps,mu),rtol=1e-6) #integration from the actual time point until the next desired time point
                    Pt_mi=odeint(evolveQuasiExactBS,P0,timevec, args=(mAC,mC,mA,dim,const,dPAC1,dPAC2,dPC1,dPC2,dPA1,dPA2,dydt,abvec,bcvec)) 
                    
                    tau=destime[h] #update the actual time point from which the next integration starts
                    b1ac[h]=sol[-1,0] #update of the species
                    b2ac[h]=sol[-1,1]
                    a1c[h]=sol[-1,2]
                    a2c[h]=sol[-1,3]
                    b1c[h]=sol[-1,4]
                    b2c[h]=sol[-1,5]
                    abc[h]=sol[-1,6]
                    b1a[h]=sol[-1,7]
                    b2a[h]=sol[-1,8]
                    c1a[h]=sol[-1,9]
                    c2a[h]=sol[-1,10]
                    bca[h]=sol[-1,11]
                    mia[h]=sol[-1,12]
                    mic[h]=sol[-1,13]
                    mi[h]=mia[h]+mic[h]                   
                    Pt_mi[-1][:dim]=Pt_mi[-1][:dim].copy()/np.sum(Pt_mi[-1][:dim])
                    Pt_mi[-1][dim+1:dim+1+dim*dim]=Pt_mi[-1][dim+1:dim+1+dim*dim].copy()/np.sum(Pt_mi[-1][dim+1:dim+1+dim*dim])
                    Pt_mi[-1][dim+1+dim*dim:]=Pt_mi[-1][dim+1+dim*dim:].copy()/np.sum(Pt_mi[-1][dim+1+dim*dim:])
                    Pt_ac[h]=Pt_mi[-1][:dim].copy()
                    Pt_c[h]=Pt_mi[-1][dim+1:dim+1+dim*dim].copy()
                    Pt_a[h]=Pt_mi[-1][dim+1+dim*dim:].copy()
                    mi_exact[h]=Pt_mi[-1][dim]
                    at[h]=a
                    bt[h]=b
                    ct[h]=c
                    h=h+1
                    if h>=len(destime)-1:
                        break
    
                y0=[sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4],sol[-1,5],sol[-1,6],sol[-1,7],sol[-1,8],sol[-1,9],sol[-1,10],sol[-1,11],sol[-1,12],sol[-1,13]]
                P0=Pt_mi[-1].copy()
    
                timevec=np.linspace(tau,t+time,5)
                sol=odeint(evolveBS,y0,timevec, args=(a,c,const,K,eps,mu),rtol=1e-6) #integration from the actual time point until the next desired time point
                Pt_mi=odeint(evolveQuasiExactBS,P0,timevec, args=(mAC,mC,mA,dim,const,dPAC1,dPAC2,dPC1,dPC2,dPA1,dPA2,dydt,abvec,bcvec))
                Pt_mi[-1][:dim]=Pt_mi[-1][:dim].copy()/np.sum(Pt_mi[-1][:dim])
                Pt_mi[-1][dim+1:dim+1+dim*dim]=Pt_mi[-1][dim+1:dim+1+dim*dim].copy()/np.sum(Pt_mi[-1][dim+1:dim+1+dim*dim])
                Pt_mi[-1][dim+1+dim*dim:]=Pt_mi[-1][dim+1+dim*dim:].copy()/np.sum(Pt_mi[-1][dim+1+dim*dim:])
                tau=t+time #update the actual time point from which the next integration starts
                
            t=t+time #update the total time 
            if index==5: #evaluation of the stochastic integral for the jumps (reaction 5)
                b1ac[h]=sol[-1,0]+(sol[-1,1]-sol[-1,0]*sol[-1,0])/sol[-1,0]
                b2ac[h]=sol[-1,1]+(2*sol[-1,1]*sol[-1,1]/sol[-1,0]-sol[-1,0]*sol[-1,1]*2)/sol[-1,0]
                a1c[h]=sol[-1,2]+(sol[-1,6]-sol[-1,2]*sol[-1,4])/sol[-1,4]
                a2c[h]=sol[-1,3]+(2*(sol[-1,3]*sol[-1,6]/sol[-1,2])-sol[-1,3]*sol[-1,4]*2)/sol[-1,4]
                b1c[h]=sol[-1,4]+(sol[-1,5]-sol[-1,4]*sol[-1,4])/sol[-1,4]
                b2c[h]=sol[-1,5]+(2*sol[-1,5]*sol[-1,5]/sol[-1,4]-2*sol[-1,5]*sol[-1,4])/sol[-1,4]
                abc[h]=sol[-1,6]+(2*sol[-1,5]*sol[-1,6]/sol[-1,4]-sol[-1,5]*sol[-1,2]-sol[-1,4]*sol[-1,6])/sol[-1,4]
                
                if sol[-1,0]==0 and sol[-1,4]!=0:
                    mic[h]=sol[-1,13]-np.log(const[4]*sol[-1,4])
                if sol[-1,0]!=0 and sol[-1,4]==0:   
                    mic[h]=sol[-1,13]+np.log(const[4]*sol[-1,0])
                if sol[-1,0]==0 and sol[-1,4]==0:
                    mic[h]=sol[-1,13]
                if sol[-1,0]!=0 and sol[-1,4]!=0:
                    mic[h]=sol[-1,13]+np.log(const[4]*sol[-1,0])-np.log(const[4]*sol[-1,4])
                sol[-1,0]=b1ac[h] 
                sol[-1,1]=b2ac[h]
                sol[-1,2]=a1c[h]
                sol[-1,3]=a2c[h]
                sol[-1,4]=b1c[h]
                sol[-1,5]=b2c[h]
                sol[-1,6]=abc[h] 
                sol[-1,13]=mic[h]
                
                Pt_ac[h]=Pt_mi[-1][:dim].copy()+dN(Pt_mi[-1][:dim].copy(),dim)
                Pt_c[h]=Pt_mi[-1][dim+1:dim+1+dim*dim].copy()+dN2D(Pt_mi[-1][dim+1:dim+1+dim*dim].copy(),dim)
                b1ac_exact=np.dot(Pt_mi[-1][:dim],[k for k in range(dim)])
                b1c_exact=mean2d(Pt_mi[-1][dim+1:dim+1+dim*dim],dim)
                if b1ac_exact==0 and b1c_exact!=0:
                    mi_exact[h]=Pt_mi[-1][dim]-np.log(const[4]*b1c_exact)
                if b1c_exact==0 and b1ac_exact!=0:
                    mi_exact[h]=Pt_mi[-1][dim]+np.log(const[4]*b1ac_exact)
                if b1c_exact==0 and b1ac_exact==0:
                    mi_exact[h]=Pt_mi[-1][dim]
                if b1c_exact!=0 and b1ac_exact!=0:
                    mi_exact[h]=Pt_mi[-1][dim]+(np.log(const[4]*b1ac_exact)-np.log(const[4]*b1c_exact))

                Pt_ac[h]=Pt_ac[h].copy()/np.sum(Pt_ac[h])
                Pt_c[h]=Pt_c[h].copy()/np.sum(Pt_c[h])
                Pt_mi[-1][:dim]=Pt_ac[h].copy()
                Pt_mi[-1][dim+1:dim+1+dim*dim]=Pt_c[h].copy()
                Pt_mi[-1][dim]=mi_exact[h]
                
            if index==1: #evaluation of the stochastic integral for the jumps (reaction 1)
                lambdaac=(mu*c*c*c/float(K*K*K+c*c*c)+eps) 
                lambdaa=(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
                b1a[h]=sol[-1,7]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(sol[-1,11]-sol[-1,9]*sol[-1,7])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
                b2a[h]=sol[-1,8]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(2*sol[-1,8]*sol[-1,11]/sol[-1,7]-2*sol[-1,8]*sol[-1,9])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
                c1a[h]=sol[-1,9]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(sol[-1,10]-sol[-1,9]*sol[-1,9])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
                c2a[h]=sol[-1,10]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(2*sol[-1,10]*sol[-1,10]/sol[-1,9]-2*sol[-1,10]*sol[-1,9])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
                bca[h]=sol[-1,11]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(2*sol[-1,10]*sol[-1,11]/sol[-1,9]-sol[-1,10]*sol[-1,7]-sol[-1,9]*sol[-1,11])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
                if lambdaac==0 and lambdaa!=0:
                    mia[h]=sol[-1,12]-np.log(lambdaa)
                if lambdaac!=0 and lambdaa==0:
                    mia[h]=sol[-1,12]+np.log(lambdaac)
                if lambdaac==0 and lambdaa==0:
                    mia[h]=sol[-1,12]
                if lambdaac!=0 and lambdaa!=0:
                    mia[h]=sol[-1,12]+np.log(lambdaac)-np.log(lambdaa)
                sol[-1,7]=b1a[h]
                sol[-1,8]=b2a[h]
                sol[-1,9]=c1a[h]
                sol[-1,10]=c2a[h]
                sol[-1,11]=bca[h]
                sol[-1,12]=mia[h]
                
                Pt_a[h]=Pt_mi[-1][dim+1+dim*dim:].copy()+dN_hill(Pt_mi[-1][dim+1+dim*dim:].copy(),dim,params)
                k1_exact=mean_hill(Pt_mi[-1][dim+1+dim*dim:],dim,params)
                if const[0]!=0 and k1_exact!=0:
                    mi_exact[h]=Pt_mi[-1][dim]+(np.log(const[0])-np.log(k1_exact))
                if const[0]==0 and k1_exact==0:
                    mi_exact[h]=Pt_mi[-1][dim]
                if const[0]!=0 and k1_exact==0:
                    mi_exact[h]=Pt_mi[-1][dim]+np.log(const[0])
                if const[0]==0 and k1_exact!=0:
                    mi_exact[h]=Pt_mi[-1][dim]-np.log(k1_exact)
                Pt_a[h]=Pt_a[h].copy()/np.sum(Pt_a[h])
                Pt_mi[-1][dim+1+dim*dim:]=Pt_a[h].copy()
                Pt_mi[-1][dim]=mi_exact[h]
                
            mi[h]=mia[h]+mic[h]
            a+=updates[index-1][0] #update of the SSA trajectories 
            b+=updates[index-1][1]
            c+=updates[index-1][2]
            updateMatrixAC(dim,a,const,mAC) #update of the matrices for the quasi-exact integration of the filtering equation 
            updateMatrixC(dim,const,mC)
            updateMatrixA(dim,a,const,mA,params)
        return at[:-1],bt[:-1],ct[:-1],mi_exact[:-1],Pt_ac,Pt_c,Pt_a,b1ac[:-1],b2ac[:-1],a1c[:-1],a2c[:-1],b1c[:-1],b2c[:-1],abc[:-1],b1a[:-1],b2a[:-1],c1a[:-1],c2a[:-1],bca[:-1],mia[:-1],mic[:-1],mi[:-1]
        
        
        
    else:
        at,bt,ct,b1ac,b2ac,a1c,a2c,b1c,b2c,abc,b1a,b2a,c1a,c2a,bca,mia,mic,mi=[np.zeros(laenge) for i in range(18)] #definition of the trajectories
        updates=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]] #stoichiometric changes
    
        t=0 #determines the whole time that has already passed 
        tau=0 #determines the actual time point for the numerical integration
        a=a0
        b=b0
        c=c0
        at[0]=a0
        bt[0]=b0
        ct[0]=c0
        t_max=destime[-1]
        h=0
        sol=np.array([[b0,b0*b0,a0,a0*a0,b0,b0*b0,a0*b0,b0,b0*b0,c0,c0*c0,b0*c0,0,0]])
        while (t<t_max): 
            
            const[0]=c*c*c/float(K*K*K+c*c*c)*mu+eps  
            propensities=[const[0], const[1]*a, const[2]*a, const[3]*b, const[4]*b, const[5]*c] #SSA 
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
                    sol=odeint(evolveBS,y0,timevec, args=(a,c,const,K,eps,mu),rtol=1e-6) #integration from the actual time point until the next desired time point
                    tau=destime[h] #update the actual time point from which the next integration starts
                    b1ac[h]=sol[-1,0] #update of the species
                    b2ac[h]=sol[-1,1]
                    a1c[h]=sol[-1,2]
                    a2c[h]=sol[-1,3]
                    b1c[h]=sol[-1,4]
                    b2c[h]=sol[-1,5]
                    abc[h]=sol[-1,6]
                    b1a[h]=sol[-1,7]
                    b2a[h]=sol[-1,8]
                    c1a[h]=sol[-1,9]
                    c2a[h]=sol[-1,10]
                    bca[h]=sol[-1,11]
                    mia[h]=sol[-1,12]
                    mic[h]=sol[-1,13]
                    mi[h]=mia[h]+mic[h]
                    at[h]=a
                    bt[h]=b
                    ct[h]=c
                    if h==250:
                        sol[-1,12]=0
                        sol[-1,13]=0
                    h=h+1
                    if h>=len(destime)-1:
                        break
                y0=[sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4],sol[-1,5],sol[-1,6],sol[-1,7],sol[-1,8],sol[-1,9],sol[-1,10],sol[-1,11],sol[-1,12],sol[-1,13]]
    
                timevec=np.linspace(tau,t+time,5)
                sol=odeint(evolveBS,y0,timevec, args=(a,c,const,K,eps,mu),rtol=1e-6)  #integration from the actual time point until the next desired time point 
                tau=t+time #update the actual time point from which the next integration starts
            t=t+time #update of the total time 
            if index==5: #evaluation of the stochastic integral for the jumps (reaction 5)
                b1ac[h]=sol[-1,0]+(sol[-1,1]-sol[-1,0]*sol[-1,0])/sol[-1,0]
                b2ac[h]=sol[-1,1]+(2*sol[-1,1]*sol[-1,1]/sol[-1,0]-sol[-1,0]*sol[-1,1]*2)/sol[-1,0]
                a1c[h]=sol[-1,2]+(sol[-1,6]-sol[-1,2]*sol[-1,4])/sol[-1,4]
                a2c[h]=sol[-1,3]+(2*(sol[-1,3]*sol[-1,6]/sol[-1,2])-sol[-1,3]*sol[-1,4]*2)/sol[-1,4]
                b1c[h]=sol[-1,4]+(sol[-1,5]-sol[-1,4]*sol[-1,4])/sol[-1,4]
                b2c[h]=sol[-1,5]+(2*sol[-1,5]*sol[-1,5]/sol[-1,4]-2*sol[-1,5]*sol[-1,4])/sol[-1,4]
                abc[h]=sol[-1,6]+(2*sol[-1,5]*sol[-1,6]/sol[-1,4]-sol[-1,5]*sol[-1,2]-sol[-1,4]*sol[-1,6])/sol[-1,4]
                
                if sol[-1,0]==0 and sol[-1,4]!=0:
                    mic[h]=sol[-1,13]-np.log(const[4]*sol[-1,4])
                if sol[-1,0]!=0 and sol[-1,4]==0:   
                    mic[h]=sol[-1,13]+np.log(const[4]*sol[-1,0])
                if sol[-1,0]==0 and sol[-1,4]==0:
                    mic[h]=sol[-1,13]
                if sol[-1,0]!=0 and sol[-1,4]!=0:
                    mic[h]=sol[-1,13]+np.log(const[4]*sol[-1,0])-np.log(const[4]*sol[-1,4])
                sol[-1,0]=b1ac[h]
                sol[-1,1]=b2ac[h]
                sol[-1,2]=a1c[h]
                sol[-1,3]=a2c[h]
                sol[-1,4]=b1c[h]
                sol[-1,5]=b2c[h]
                sol[-1,6]=abc[h] 
                sol[-1,13]=mic[h]
    
                    
            if index==1: #evaluation of the stochastic integral for the jumps (reaction 1)
                lambdaac=(mu*c*c*c/float(K*K*K+c*c*c)+eps) 
                lambdaa=(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
                b1a[h]=sol[-1,7]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(sol[-1,11]-sol[-1,9]*sol[-1,7])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
                b2a[h]=sol[-1,8]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(2*sol[-1,8]*sol[-1,11]/sol[-1,7]-2*sol[-1,8]*sol[-1,9])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
                c1a[h]=sol[-1,9]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(sol[-1,10]-sol[-1,9]*sol[-1,9])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
                c2a[h]=sol[-1,10]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(2*sol[-1,10]*sol[-1,10]/sol[-1,9]-2*sol[-1,10]*sol[-1,9])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
                bca[h]=sol[-1,11]+(3*mu*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])-3*mu*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]*sol[-1,9]/((K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])*(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])))*(2*sol[-1,10]*sol[-1,11]/sol[-1,9]-sol[-1,10]*sol[-1,7]-sol[-1,9]*sol[-1,11])/(mu*sol[-1,9]*sol[-1,9]*sol[-1,9]/(K*K*K+sol[-1,9]*sol[-1,9]*sol[-1,9])+eps)
                if lambdaac==0 and lambdaa!=0:
                    mia[h]=sol[-1,12]-np.log(lambdaa)
                if lambdaac!=0 and lambdaa==0:
                    mia[h]=sol[-1,12]+np.log(lambdaac)
                if lambdaac==0 and lambdaa==0:
                    mia[h]=sol[-1,12]
                if lambdaac!=0 and lambdaa!=0:
                    mia[h]=sol[-1,12]+np.log(lambdaac)-np.log(lambdaa)
                sol[-1,7]=b1a[h]
                sol[-1,8]=b2a[h]
                sol[-1,9]=c1a[h]
                sol[-1,10]=c2a[h]
                sol[-1,11]=bca[h]
                sol[-1,12]=mia[h]
                
            mi[h]=mia[h]+mic[h]
            a+=updates[index-1][0] #update of the SSA trajectories
            b+=updates[index-1][1]
            c+=updates[index-1][2]
        return at[:-1],bt[:-1],ct[:-1],b1ac[:-1],b2ac[:-1],a1c[:-1],a2c[:-1],b1c[:-1],b2c[:-1],abc[:-1],b1a[:-1],b2a[:-1],c1a[:-1],c2a[:-1],bca[:-1],mia[:-1],mic[:-1],mi[:-1]

def parallelisationBS(core,MC,iniconds,const,params,laenge,timevec,exact=False,dim=50):
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
    if exact==True:
        atdata=[] #definition of the final arrays
        btdata=[]
        ctdata=[]
        b1acdata=[]
        b2acdata=[]
        b1cdata=[]
        b2cdata=[]
        a1cdata=[]
        a2cdata=[]
        abcdata=[]
        b1adata=[]
        b2adata=[]
        c1adata=[]
        c2adata=[]
        miadata=[]
        micdata=[]
        midata=[]
        p_acdata=[]
        p_cdata=[]
        p_adata=[]
        miexactdata=[]
        with Pool(core) as pool: #parallelisation
            results = pool.map(partial(computeTrajectoryBS,iniconds=iniconds,const=const,params=params,laenge=laenge,destime=timevec), [k for k in range(MC)])
            for r in results:
                at_temp,bt_temp,ct_temp,miexacttemp,pt_ac_temp,pt_c_temp,pt_a_temp,b1ac_temp,b2ac_temp,a1c_temp,a2c_temp,b1c_temp,b2c_temp,abc_temp,b1a_temp,b2a_temp,c1a_temp,c2a_temp,bca_temp,mia_temp,mic_temp,mi_temp = r
                atdata.append(at_temp) #collecting all the single trajectories 
                btdata.append(bt_temp)
                ctdata.append(ct_temp)
                b1acdata.append(b1ac_temp)
                b2acdata.append(b2ac_temp)
                b1cdata.append(b1c_temp)
                b1adata.append(b1a_temp)
                b2adata.append(b2a_temp)
                b2cdata.append(b2c_temp)
                a1cdata.append(a1c_temp)
                a2cdata.append(a2c_temp)
                abcdata.append(abc_temp)
                c1adata.append(c1a_temp)
                c2adata.append(c2a_temp)
                miadata.append(mia_temp)
                micdata.append(mic_temp)
                midata.append(mi_temp)
                p_acdata.append(pt_ac_temp)
                p_cdata.append(pt_c_temp)
                p_adata.append(pt_a_temp)
                miexactdata.append(miexacttemp)
            return atdata,btdata,ctdata,p_acdata,p_cdata,p_adata,miexactdata,b1acdata,b2acdata,a1cdata,a2cdata,b1cdata,b2cdata,abcdata,b1adata,b2adata,c1adata,c2adata,miadata,micdata,midata
    else:
        atdata=[] #definition of the final arrays
        btdata=[]
        ctdata=[]
        b1acdata=[]
        b2acdata=[]
        b1cdata=[]
        b2cdata=[]
        a1cdata=[]
        a2cdata=[]
        abcdata=[]
        b1adata=[]
        b2adata=[]
        c1adata=[]
        c2adata=[]
        miadata=[]
        micdata=[]
        midata=[]
        with Pool(core) as pool: #parallelisation
            results = pool.map(partial(computeTrajectoryBS,iniconds=iniconds,const=const,params=params,laenge=laenge,destime=timevec), [k for k in range(MC)])
            for r in results:
                at_temp,bt_temp,ct_temp,b1ac_temp,b2ac_temp,a1c_temp,a2c_temp,b1c_temp,b2c_temp,abc_temp,b1a_temp,b2a_temp,c1a_temp,c2a_temp,bca_temp,mia_temp,mic_temp,mi_temp = r
                atdata.append(at_temp) #collecting all the single trajectories 
                btdata.append(bt_temp)
                ctdata.append(ct_temp)
                b1acdata.append(b1ac_temp)
                b2acdata.append(b2ac_temp)
                b1cdata.append(b1c_temp)
                b1adata.append(b1a_temp)
                b2adata.append(b2a_temp)
                b2cdata.append(b2c_temp)
                a1cdata.append(a1c_temp)
                a2cdata.append(a2c_temp)
                abcdata.append(abc_temp)
                c1adata.append(c1a_temp)
                c2adata.append(c2a_temp)
                miadata.append(mia_temp)
                micdata.append(mic_temp)
                midata.append(mi_temp)
            return atdata,btdata,ctdata,b1acdata,b2acdata,a1cdata,a2cdata,b1cdata,b2cdata,abcdata,b1adata,b2adata,c1adata,c2adata,miadata,micdata,midata
    
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
    


    
    
    
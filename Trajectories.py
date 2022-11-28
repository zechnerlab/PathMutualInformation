#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 09:40:13 2022

@author: moor
"""

import numpy as np
from functions import evolveFF3Node,evolveFF2Node,updateMatrixAC,updateMatrixC ,updateMatrixB,dN,dN2D,mean2d
import random as rnd
from scipy.integrate import odeint
from exactsolutions import evolveQuasiExactAC,evolveQuasiExactAB
from functools import partial
from multiprocessing import Pool

def computeTrajectoryAC(n,exact,iniconds,const,dim,laenge,destime):
    '''
    Integrates path mutual information and the required conditional moments of B given A and C and of A and B given C over time 
    and determines the SSA-trajectory of the species

    Parameters
    ----------
    n : integer; for multiprocessing
    exact : keyword; determines if the exact integration of the filtering equation is supposed to be performed or not
    iniconds : list with three integer-entries; initial conditions of our species
    const : list; reaction constants of the system 
    dim : integer; dimension lattice for integrating the filtering equation
    laenge : integer, length of the trajectories
    destime : array of float; time vector of the integration

    Returns 
    -------
    The output contains several objects which are different depending on the assignment of 'exact'.
    if exact==True: 
        trajectory of A, of B, probability distribution conditional on A and C (array of dimension dim), 
        probability distribution conditional on C (array of dimension dimxdim), path mutual information, pmi squared, 
        trajectory of the expected value of B conditional on A and C, of B conditional on C, path mutual information via moment closure, pmi via closure squared
    if exact==False:
        trajectory of A, of B, trajectory the expected value of of B conditional on A and C, of B conditional on C, path mutual information via moment closure, pmi via closure squared
        
    '''
    abvec=np.zeros(dim*dim) #defines the lattice of the copy numbers of species A and B
    for i in range(dim*dim):
        abvec[i]=i%dim
    updates=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]] #stoichiometric changes 
    a0,b0,c0=iniconds
    if exact==True:
        Pt_mi=np.zeros((1,dim*dim+dim+1)) #array that will contain the conditional probability distributions and the mutual information, dimension will change
        dydt=np.zeros(dim*dim+dim+1) #for the numerical integration
        Pt_mi[0][b0]=1
        Pt_mi[0][dim+1+dim*a0+b0]=1
        a_g,b_g,c_g,mi,b1_AC,b2_AC,b1_C,b2_C,a1_C,a2_C,ab_C,mi_closured=[np.zeros(laenge) for i in range(12)]
        dPAC1,dPAC2=[np.zeros(dim) for i in range(2)] #for the numerical integration
        dPC1,dPC2=[np.zeros(dim*dim) for i in range(2)] #for the numerical integration
    
        t=0 #determines the whole time that has already passed 
        tau=0 #determines the actual time point for the numerical integration
        Pt_ac=np.zeros((laenge,dim)) #array that will contain all the probability distributions conditional on A and C of the different time points
        Pt_c=np.zeros((laenge,dim*dim)) #array that will contain all the probability distributions conditional on C of the different time points
        a=a0
        b=b0
        c=c0
        a_g[0]=a0
        b_g[0]=b0
        c_g[0]=c0
        t_max=destime[-1]
        mAC=np.zeros((dim,dim)) #integration matrix for Pt_ac
        updateMatrixAC(dim,a,const,mAC)
        mC=np.zeros((dim*dim,dim*dim)) #integration matrix for Pt_c
        updateMatrixC(dim,const,mC)
        h=0
        sol=np.array([[b0, b0*b0, a0,b0,a0*a0,b0*b0,a0*b0,0]])
        propensities=[const[0], const[1]*a, const[2]*a, const[3]*b, const[4]*b, const[5]*c]
        
        while (t<t_max): 
            reacsum=sum(propensities) #SSA 
            time=-np.log(rnd.uniform(0.0,1.0))/reacsum     #SSA 
            RV=rnd.uniform(0.0,1.0)*reacsum      #SSA 
            index=1 #SSA 
            value=propensities[0] #SSA 
            while value<RV: #SSA 
                index+=1 #SSA 
                value+=propensities[index-1]#SSA 
            propensities=[const[0], const[1]*a, const[2]*a, const[3]*b, const[4]*b, const[5]*c] #SSA 
            
            if h<len(destime): #step-wise integration algorithm of the conditional probability distributions and the path mutual information 
                while destime[h]<(t+time): 
                    y0=[sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4],sol[-1,5],sol[-1,6],sol[-1,7]]
                    P0=Pt_mi[-1].copy()
                    timevec=np.linspace(tau,destime[h],5) #integration interval
                    sol=odeint(evolveFF3Node,y0,timevec, args=(a,const)) #integration from the actual time point until the next desired time point
                    Pt_mi=odeint(evolveQuasiExactAC,P0,timevec, args=(mAC,mC,a,dim,const,dPAC1,dPAC2,dPC1,dPC2,dydt,abvec))
                    
                    tau=destime[h] #update the actual time point from which the next integration starts
                    b1_AC[h]=sol[-1,0] #update of the species
                    b2_AC[h]=sol[-1,1]
                    a1_C[h]=sol[-1,2]
                    a2_C[h]=sol[-1,4]
                    b1_C[h]=sol[-1,3]
                    b2_C[h]=sol[-1,5]
                    ab_C[h]=sol[-1,6]
                    mi_closured[h]=sol[-1,7]
                    Pt_mi[-1][:dim]=Pt_mi[-1][:dim].copy()/np.sum(Pt_mi[-1][:dim])
                    Pt_mi[-1][dim+1:]=Pt_mi[-1][dim+1:].copy()/np.sum(Pt_mi[-1][dim+1:])
                    Pt_ac[h]=Pt_mi[-1][:dim].copy()
                    Pt_c[h]=Pt_mi[-1][dim+1:].copy()
                    mi[h]=Pt_mi[-1][dim]
                    a_g[h]=a
                    b_g[h]=b
                    h=h+1
                    if h>=len(destime)-1:
                        break
    
                y0=[sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4],sol[-1,5],sol[-1,6],sol[-1,7]]
                P0=Pt_mi[-1].copy()
    
                timevec=np.linspace(tau,t+time,5)
                sol=odeint(evolveFF3Node,y0,timevec, args=(a,const)) #integration from the actual time point until the next desired time point
                Pt_mi=odeint(evolveQuasiExactAC,P0,timevec, args=(mAC,mC,a,dim,const,dPAC1,dPAC2,dPC1,dPC2,dydt,abvec))
                Pt_mi[-1][:dim]=Pt_mi[-1][:dim].copy()/np.sum(Pt_mi[-1][:dim])
                Pt_mi[-1][dim+1:]=Pt_mi[-1][dim+1:].copy()/np.sum(Pt_mi[-1][dim+1:])
                tau=t+time #update the actual time point from which the next integration starts
            t=t+time #update the total time 
            if index==5: #evaluation of the stochastic integral for the jumps (reaction 5)
                b1_AC[h]=(sol[-1,1]-sol[-1,0]*sol[-1,0])/sol[-1,0]+sol[-1,0]
                b2_AC[h]=sol[-1,1]+(2*sol[-1,1]*sol[-1,1]/sol[-1,0]-sol[-1,0]*sol[-1,1]*2)/sol[-1,0] 
                b1_C[h]=sol[-1,3]+(sol[-1,5]-sol[-1,3]*sol[-1,3])/sol[-1,3]
                b2_C[h]=sol[-1,5]+(2*(sol[-1,5]*sol[-1,5]/sol[-1,3])-sol[-1,3]*sol[-1,5]*2)/sol[-1,3]
                a1_C[h]=sol[-1,2]+(sol[-1,6]-sol[-1,2]*sol[-1,3])/sol[-1,3]
                a2_C[h]=sol[-1,4]+(2*(sol[-1,4]*sol[-1,6]/sol[-1,2])-sol[-1,4]*sol[-1,3]*2)/sol[-1,3]
                ab_C[h]=sol[-1,6]+(2*(sol[-1,5]*sol[-1,6]/sol[-1,3])-sol[-1,5]*sol[-1,2]-sol[-1,3]*sol[-1,6])/sol[-1,3]
                if sol[-1,0]==0 and sol[-1,3]!=0:
                    mi_closured[h]=sol[-1,7]+(-np.log(const[4]*sol[-1,3]))
                if sol[-1,0]!=0 and sol[-1,3]==0:
                    mi_closured[h]=sol[-1,7]+(np.log(const[4]*sol[-1,0]))
                if sol[-1,0]==0 and sol[-1,3]==0:
                    mi_closured[h]=sol[-1,7]
                if sol[-1,0]!=0 and sol[-1,3]!=0:
                    mi_closured[h]=sol[-1,7]+(np.log(const[4]*sol[-1,0])-np.log(const[4]*sol[-1,3]))
                sol[-1,7]=mi_closured[h]
                sol[-1,0]=b1_AC[h]
                sol[-1,1]=b2_AC[h]
                sol[-1,3]=b1_C[h]
                sol[-1,5]=b2_C[h]
                sol[-1,2]=a1_C[h]
                sol[-1,4]=a2_C[h]
                sol[-1,6]=ab_C[h]
                
                Pt_ac[h]=Pt_mi[-1][:dim].copy()+dN(Pt_mi[-1][:dim].copy(),dim)
                Pt_c[h]=Pt_mi[-1][dim+1:].copy()+dN2D(Pt_mi[-1][dim+1:].copy(),dim)
                bac=np.dot(Pt_mi[-1][:dim],[k for k in range(dim)])
                bc=mean2d(Pt_mi[-1][dim+1:],dim)
                if bac==0 and bc!=0:
                    mi[h]=Pt_mi[-1][dim]-np.log(const[4]*bc)
                if bc==0 and bac!=0:
                    mi[h]=Pt_mi[-1][dim]+np.log(const[4]*bac)
                if bc==0 and bac==0:
                    mi[h]=Pt_mi[-1][dim]
                if bc!=0 and bac!=0:
                    mi[h]=Pt_mi[-1][dim]+(np.log(const[4]*bac)-np.log(const[4]*bc))
                tau=t
                Pt_ac[h]=Pt_ac[h].copy()/np.sum(Pt_ac[h])
                Pt_c[h]=Pt_c[h].copy()/np.sum(Pt_c[h])
                Pt_mi[-1][:dim]=Pt_ac[h].copy()
                Pt_mi[-1][dim+1:]=Pt_c[h].copy()
                Pt_mi[-1][dim]=mi[h]
            a+=updates[index-1][0] #update of the SSA trajectories 
            b+=updates[index-1][1]
            c+=updates[index-1][2]
            updateMatrixAC(dim,a,const,mAC) #update of the matrices for the quasi-exact integration of the filtering equation 
            updateMatrixC(dim,const,mC)
        return a_g,b_g,Pt_ac,Pt_c,mi,mi*mi,b1_AC,b1_C,mi_closured,mi_closured*mi_closured
    else: #if exact==False, we perform the calculation of the path mutual information analogously but only for the system with moment closure. 
        a_g,b_g,c_g,b1_AC,b2_AC,b1_C,b2_C,a1_C,a2_C,ab_C,mi_closured=[np.zeros(laenge) for i in range(11)]
    
        t=0
        tau=0
        a=a0
        b=b0
        c=c0
        a_g[0]=a0
        b_g[0]=b0
        c_g[0]=c0
        t_max=destime[-1]
        h=0
        sol=np.array([[b0, b0*b0, a0,b0,a0*a0,b0*b0,a0*b0,0]])
        
        while (t<t_max):
            propensities=[const[0], const[1]*a, const[2]*a, const[3]*b, const[4]*b, const[5]*c]
            reacsum=sum(propensities)
            time=-np.log(rnd.uniform(0.0,1.0))/reacsum     
            RV=rnd.uniform(0.0,1.0)*reacsum      
            index=1
            value=propensities[0]
            while value<RV:
                index+=1
                value+=propensities[index-1]  
            
            if h<len(destime):
                while destime[h]<(t+time): 
                    y0=[sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4],sol[-1,5],sol[-1,6],sol[-1,7]]
                    timevec=np.linspace(tau,destime[h],5)
                    sol=odeint(evolveFF3Node,y0,timevec, args=(a,const))
                    tau=destime[h]
                    b1_AC[h]=sol[-1,0]
                    b2_AC[h]=sol[-1,1]
                    a1_C[h]=sol[-1,2]
                    a2_C[h]=sol[-1,4]
                    b1_C[h]=sol[-1,3]
                    b2_C[h]=sol[-1,5]
                    ab_C[h]=sol[-1,6]
                    mi_closured[h]=sol[-1,7]
                    a_g[h]=a
                    b_g[h]=b
                    h=h+1
                    if h>=len(destime)-1:
                        break
    
                y0=[sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4],sol[-1,5],sol[-1,6],sol[-1,7]]
    
                timevec=np.linspace(tau,t+time,5)
                sol=odeint(evolveFF3Node,y0,timevec, args=(a,const))
                tau=t+time
            t=t+time
            if index==5:
                b1_AC[h]=(sol[-1,1]-sol[-1,0]*sol[-1,0])/sol[-1,0]+sol[-1,0]
                b2_AC[h]=sol[-1,1]+(2*sol[-1,1]*sol[-1,1]/sol[-1,0]-sol[-1,0]*sol[-1,1]*2)/sol[-1,0] 
                b1_C[h]=sol[-1,3]+(sol[-1,5]-sol[-1,3]*sol[-1,3])/sol[-1,3]
                b2_C[h]=sol[-1,5]+(2*(sol[-1,5]*sol[-1,5]/sol[-1,3])-sol[-1,3]*sol[-1,5]*2)/sol[-1,3] 
                a1_C[h]=sol[-1,2]+(sol[-1,6]-sol[-1,2]*sol[-1,3])/sol[-1,3]
                a2_C[h]=sol[-1,4]+(2*(sol[-1,4]*sol[-1,6]/sol[-1,2])-sol[-1,4]*sol[-1,3]*2)/sol[-1,3] 
                ab_C[h]=sol[-1,6]+(2*(sol[-1,5]*sol[-1,6]/sol[-1,3])-sol[-1,5]*sol[-1,2]-sol[-1,3]*sol[-1,6])/sol[-1,3]
                if sol[-1,0]==0 and sol[-1,3]!=0:
                    mi_closured[h]=sol[-1,7]+(-np.log(const[4]*sol[-1,3]))
                if sol[-1,0]!=0 and sol[-1,3]==0:
                    mi_closured[h]=sol[-1,7]+(np.log(const[4]*sol[-1,0]))
                if sol[-1,0]==0 and sol[-1,3]==0:
                    mi_closured[h]=sol[-1,7]
                if sol[-1,0]!=0 and sol[-1,3]!=0:
                    mi_closured[h]=sol[-1,7]+(np.log(const[4]*sol[-1,0])-np.log(const[4]*sol[-1,3]))
                sol[-1,7]=mi_closured[h]
                sol[-1,0]=b1_AC[h]
                sol[-1,1]=b2_AC[h]
                sol[-1,3]=b1_C[h]
                sol[-1,5]=b2_C[h]
                sol[-1,2]=a1_C[h]
                sol[-1,4]=a2_C[h]
                sol[-1,6]=ab_C[h]
            a+=updates[index-1][0]
            b+=updates[index-1][1]
            c+=updates[index-1][2]
        return a_g,b_g,b1_AC,b1_C,mi_closured,mi_closured*mi_closured
    
def computeTrajectoryAB(n,exact,iniconds,const,dim,laenge,destime):
    '''
    Integrates path mutual information and the required conditional moments of A given B over time 
    and determines the SSA-trajectory of the species

    Parameters
    ----------
    n : integer; for multiprocessing
    exact : keyword; determines if the exact integration of the filtering equation is supposed to be performed or not
    iniconds : list with two integer-entries; initial conditions of our species
    const : list; reaction constants of the system 
    dim : integer; dimension lattice for integrating the filtering equation
    laenge : integer, length of the trajectories
    destime : array of float; time vector of the integration

    Returns 
    -------
    The output contains several objects which are different depending on the assignment of 'exact'.
    if exact==True: 
        trajectory of A, of B, probability distribution conditional B (array of dimension dim), 
        path mutual information, pmi squared, trajectory of the expected value of A conditional on B, its second moment, path mutual information via moment closure, pmi via closure squared
    if exact==False:
        trajectory of A, of B, trajectory of the expected value of A conditional on B, its second moment, path mutual information via moment closure, pmi via closure squared
        
    '''
    updates=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]]
    a0,b0=iniconds
    if exact==True:
        Pt_mi=np.zeros((1,dim+1))
        Pt_mi[0][a0]=1
        dydt=np.zeros(dim+1)
        dP=np.zeros(dim)
        a_g,b_g,mi,a1_B,a2_B,mi_closured=[np.zeros(laenge) for i in range(6)]
        a1_B[0]=a0
        a2_B[0]=a0*a0
        t=0
        tau=0
        Pt_tr=np.zeros((laenge,dim))
        a=a0
        b=b0
        a_g[0]=a0
        b_g[0]=b0
        t_max=destime[-1]
        mB=updateMatrixB(dim,const[0],const[1],const[2])
        h=0
        sol=np.array([[a0, a0*a0, 0]])
        while (t<t_max):
            propensities=[const[0], const[1]*a, const[2]*a, const[3]*b]
            reacsum=sum(propensities)
            time=-np.log(rnd.uniform(0.0,1.0))/reacsum     
            RV=rnd.uniform(0.0,1.0)*reacsum      
            index=1
            value=propensities[0]
            while value<RV:
                index+=1
                value+=propensities[index-1]  
                
            if h<len(destime):
                while destime[h]<(t+time):
                    y0=[sol[-1,0],sol[-1,1],sol[-1,2]]
                    P0=Pt_mi[-1].copy()
                    timevec=np.linspace(tau,destime[h],5)
                    sol=odeint(evolveFF2Node,y0,timevec, args=(a,const))
                    Pt_mi=odeint(evolveQuasiExactAB,P0,timevec, args=(mB,a,dim,const,dP,dydt))
                    tau=destime[h]
                    a1_B[h]=sol[-1,0]
                    a2_B[h]=sol[-1,1]
                    mi_closured[h]=sol[-1,2]
                    Pt_mi[-1][:-1]=Pt_mi[-1][:-1].copy()/np.sum(Pt_mi[-1][:-1])
                    Pt_tr[h]=Pt_mi[-1][:-1].copy()
                    mi[h]=Pt_mi[-1][-1]
                    a_g[h]=a
                    b_g[h]=b
                    h=h+1
                    if h>=len(destime)-1:
                        break
                y0=[sol[-1,0],sol[-1,1],sol[-1,2]]
                P0=Pt_mi[-1].copy()
                timevec=np.linspace(tau,t+time,5)
                sol=odeint(evolveFF2Node,y0,timevec, args=(a,const))
                Pt_mi=odeint(evolveQuasiExactAB,P0,timevec, args=(mB,a,dim,const,dP,dydt))
                Pt_mi[-1][:-1]=Pt_mi[-1][:-1].copy()/np.sum(Pt_mi[-1][:-1])
                tau=t+time
            t=t+time
            if index==3:
                if a==0 and sol[-1,0]!=0:
                    mi_closured[h]=sol[-1,2]+(-np.log(const[2]*sol[-1,0]))
                if a!=0 and sol[-1,0]==0:
                    mi_closured[h]=sol[-1,2]+(np.log(const[2]*a))
                if a==0 and sol[-1,0]==0:
                    mi_closured[h]=sol[-1,2]
                if a!=0 and sol[-1,0]!=0:
                    mi_closured[h]=sol[-1,2]+(np.log(const[2]*a)-np.log(const[2]*sol[-1,0]))
                a1_B[h]=sol[-1,0]+(sol[-1,1]-sol[-1,0]*sol[-1,0])/sol[-1,0]
                a2_B[h]=sol[-1,1]+(2*(sol[-1,1]/(sol[-1,0]*sol[-1,0]))*(sol[-1,1]-sol[-1,0]*sol[-1,0]))
                Pt_tr[h]=Pt_mi[-1][:-1].copy()+dN(Pt_mi[-1][:-1].copy(),dim)
                if np.dot(Pt_mi[-1][:-1],[k for k in range(dim)])==0 and a!=0:
                    mi[h]=Pt_mi[-1][-1]+np.log(const[2]*a)
                if a==0 and np.dot(Pt_mi[-1][:-1],[k for k in range(dim)])!=0:
                    mi[h]=Pt_mi[-1][-1]-np.log(const[2]*(np.dot(Pt_mi[-1][:-1],[k for k in range(dim)])))
                if a==0 and np.dot(Pt_mi[-1][:-1],[k for k in range(dim)])==0:
                    mi[h]=Pt_mi[-1][-1]
                if a!=0 and np.dot(Pt_mi[-1][:-1],[k for k in range(dim)])!=0:
                    mi[h]=Pt_mi[-1][-1]+(np.log(const[2]*a)-np.log(const[2]*(np.dot(Pt_mi[-1][:-1],[k for k in range(dim)]))))
                tau=t
                Pt_tr[h]=Pt_tr[h].copy()/np.sum(Pt_tr[h])
                sol[-1,2]=mi_closured[h]
                sol[-1,0]=a1_B[h]
                sol[-1,1]=a2_B[h]
                Pt_mi[-1][:-1]=Pt_tr[h].copy()
                Pt_mi[-1][-1]=mi[h]
            a+=updates[index-1][0]
            b+=updates[index-1][1]
        return a_g,b_g,Pt_tr,mi,mi*mi,a1_B,a2_B,mi_closured,mi_closured*mi_closured
    else:
        a_g,b_g,a1_B,a2_B,mi_closured=[np.zeros(laenge) for i in range(5)]
        a1_B[0]=a0
        a2_B[0]=a0*a0
        t=0
        tau=0
        a=a0
        b=b0
        a_g[0]=a0
        b_g[0]=b0
        t_max=destime[-1]
        h=0
        sol=np.array([[a0, a0*a0, 0]])
        propensities=[const[0], const[1]*a, const[2]*a, const[3]*b]
        while (t<t_max):
            reacsum=sum(propensities)
            time=-np.log(rnd.uniform(0.0,1.0))/reacsum     
            RV=rnd.uniform(0.0,1.0)*reacsum      
            index=1
            value=propensities[0]
            while value<RV:
                index+=1
                value+=propensities[index-1]  
            propensities=[const[0], const[1]*a, const[2]*a, const[3]*b]
            if h<len(destime):
                while destime[h]<(t+time):
                    timevec=np.linspace(tau,destime[h],5)
                    sol=odeint(evolveFF2Node,y0,timevec, args=(a,const))
                    tau=destime[h]
                    a1_B[h]=sol[-1,0]
                    a2_B[h]=sol[-1,1]
                    mi_closured[h]=sol[-1,2]
                    a_g[h]=a
                    b_g[h]=b
                    h=h+1
                    if h>=len(destime)-1:
                        break
                timevec=np.linspace(tau,t+time,5)
                sol=odeint(evolveFF2Node,y0,timevec, args=(a,const))
                tau=t+time
            t=t+time
            if index==3:
                if a==0 and sol[-1,0]!=0:
                    mi_closured[h]=sol[-1,2]+(-np.log(const[2]*sol[-1,0]))
                if a!=0 and sol[-1,0]==0:
                    mi_closured[h]=sol[-1,2]+(np.log(const[2]*a))
                if a==0 and sol[-1,0]==0:
                    mi_closured[h]=sol[-1,2]
                if a!=0 and sol[-1,0]!=0:
                    mi_closured[h]=sol[-1,2]+(np.log(const[2]*a)-np.log(const[2]*sol[-1,0]))
                a1_B[h]=sol[-1,0]+(sol[-1,1]-sol[-1,0]*sol[-1,0])/sol[-1,0]
                a2_B[h]=sol[-1,1]+(2*(sol[-1,1]/(sol[-1,0]*sol[-1,0]))*(sol[-1,1]-sol[-1,0]*sol[-1,0]))
                tau=t
                sol[-1,2]=mi_closured[h]
                sol[-1,0]=a1_B[h]
                sol[-1,1]=a2_B[h]
            a+=updates[index-1][0]
            b+=updates[index-1][1]
        return a_g,b_g,a1_B,a2_B,mi_closured,mi_closured*mi_closured
    
def parallelisationAC(core,MC,exact,iniconds,const,dim,laenge,timevec):
    '''
    Parallelisation of the Monte Carlo integration of the path mutual information between A and C

    Parameters
    ----------
    core : integer; number of cores of the computer that are used for the calculation
    MC : integer; sample size of the Monte Carlo integration
    exact : keyword; determines if the exact integration of the filtering equation is supposed to be performed or not
    iniconds : list with three integer-entries; initial conditions of our species
    const : list; reaction constants of the system
    dim : integer; dimension lattice for integrating the filtering equation
    laenge : integer, length of the trajectories
    timevec : array of float; time vector of the integration

    Returns
    -------
    The output contains several objects which are different depending on the assignment of 'exact'. The outputs here are all Monte Carlo averages.
    if exact==True:
        species A, species B, the moment of B conditional on A and C, the moment of B conditional on C, the path mutual information, pmi squared, 
        the path mutual information rate, all the single realisations of the path mutual information (array of dimension MCxlaenge), the closed moment of B conditional on A and C, 
        the closed moment of B conditional on C, the path mutual information via closure, the pmi via closure squared, the path mutual information rate via closure, 
        all the single realisations of the path mututal information via moment closure (array of dimension MCxlaenge)
    if exact==False: 
        species A, species B, the closed moment of B conditional on A and C, the closed moment of B conditional on C, 
        the path mutual information via closure, the pmi via closure squared, the path mutual information rate via closure, 
        all the single realisations of the path mututal information via moment closure (array of dimension MCxlaenge)
    '''
    a_g,b_g,mi,b1ac,b1c,mi_closured,misquared,mi_closured_squared,meanac,meanc=[np.zeros(laenge) for i in range(10)] #definition of the final arrays
    miclodata=[] #array collecting all the single trajectories of the path mutual information via moment closure
    miexdata=[] #array collecting all the single trajectories of the path mutual information via quasi-exact integration 
    probc=np.zeros((laenge,dim*dim)) #conditional probability of A and B given C
    probac=np.zeros((laenge,dim)) #conditional probability of B given A and C
    
    with Pool(core) as pool: #parallelisation
        results = pool.map(partial(computeTrajectoryAC,exact=exact,iniconds=iniconds,const=const,dim=dim,laenge=laenge,destime=timevec), [k for k in range(MC)])
        if exact==True: #Monte Carlo average for the quasi exact method
            for r in results:
                ag,bg,probgac,probgc,mig,misqg,b1acg,b1cg,miclog,miclosqg = r
                meanacg,meancg=[np.zeros(laenge) for i in range(2)]
                a_g += ag
                b_g += bg
                probac += probgac
                probc += probgc
                mi += mig
                b1ac += b1acg
                b1c += b1cg
                mi_closured += miclog
                miclodata.append(mi_closured[:-1])
                miexdata.append(mig[:-1])
                misquared += misqg
                mi_closured_squared += miclosqg 
                for i in range(laenge):
                    meanacg[i]=np.dot(probgac[i],[k for k in range(dim)])
                    meancg[i]=mean2d(probgc[i],dim)
                meanac += meanacg
                meanc += meancg
            a_g=a_g/MC
            b_g=b_g/MC
            mi=mi/MC
            probac=probac/MC
            probc=probc/MC
            b1ac=b1ac/MC
            b1c=b1c/MC
            mi_closured=mi_closured/MC
            misquared=misquared/MC
            mi_closured_squared=mi_closured_squared/MC
            meanac=meanac/MC
            meanc=meanc/MC
            rate=mi[:-1]/timevec #this might give a warning because timevec[0]=0
            rate_closured=mi_closured[:-1]/timevec #this might give a warning because timevec[0]=0
            return a_g[:-1],b_g[:-1],meanac[:-1],meanc[:-1],mi[:-1],misquared[:-1],rate,miexdata[:-1],b1ac[:-1],b1c[:-1],mi_closured[:-1],mi_closured_squared[:-1],rate_closured,miclodata
        else: #Monte Carlo average for the system with moment closure 
            for r in results:
                ag,bg,b1acg,b1cg,miclog,miclosqg = r
                a_g += ag
                b_g += bg
                b1ac += b1acg
                b1c += b1cg
                mi_closured += miclog
                mi_closured_squared += miclosqg 
                miclodata.append(miclog[:-1])
            a_g=a_g/MC
            b_g=b_g/MC
            b1ac=b1ac/MC
            b1c=b1c/MC
            mi_closured=mi_closured/MC
            mi_closured_squared=mi_closured_squared/MC
            rate_closured=mi_closured[:-1]/timevec #this might give a warning because timevec[0]=0
            return a_g[:-1],b_g[:-1],b1ac[:-1],b1c[:-1],mi_closured[:-1],mi_closured_squared[:-1],rate_closured,miclodata
        
def parallelisationAB(core,MC,exact,iniconds,const,dim,laenge,timevec):
    '''
    Parallelisation of the Monte Carlo integration of the path mutual information between A and B

    Parameters
    ----------
    core : integer; number of cores of the computer that are used for the calculation
    MC : integer; sample size of the Monte Carlo integration
    exact : keyword; determines if the exact integration of the filtering equation is supposed to be performed or not
    iniconds : list with two integer-entries; initial conditions of our species
    const : list; reaction constants of the system
    dim : integer; dimension lattice for integrating the filtering equation
    laenge : integer, length of the trajectories
    timevec : array of float; time vector of the integration

    Returns
    -------
    The output contains several objects which are different depending on the assignment of 'exact'. The outputs here are all Monte Carlo averages.
    if exact==True:
        species A, species B, the moment of A conditional on B, its second moment, the path mutual information, pmi squared, 
        the path mutual information rate, the closed moment of A conditional on B, its second moment, 
        the path mutual information via closure, the pmi via closure squared, the path mutual information rate via closure
    if exact==False: 
        species A, species B, the closed moment of A conditional on B, its second moment, 
        the path mutual information via closure, the pmi via closure squared, the path mutual information rate via closure
    '''
    a_g,b_g,mi,a1b,a2b,mi_closured,misquared,mi_closured_squared,mean,secmom=[np.zeros(laenge) for i in range(10)]
    prob=np.zeros((laenge,dim))
    with Pool(core) as pool:
        results = pool.map(partial(computeTrajectoryAB,exact=exact,iniconds=iniconds,const=const,dim=dim,laenge=laenge,destime=timevec), [k for k in range(MC)])
        if exact==True:
            for r in results:
                ag,bg,probg,mig,misqg,a1bg,a2bg,miclog,miclosqg = r
                meang=np.zeros(laenge)
                secmomg=np.zeros(laenge)
                for i in range(laenge):
                    meang[i]=np.dot(probg[i],[k for k in range(dim)]) 
                    secmomg[i]=np.dot(probg[i],[k*k for k in range(dim)])
                mean += meang
                secmom += secmomg
                a_g += ag
                b_g += bg
                prob += probg
                mi += mig
                a1b += a1bg
                a2b += a2bg
                mi_closured += miclog
                misquared += misqg
                mi_closured_squared += miclosqg 
            a_g=a_g/MC
            b_g=b_g/MC
            prob=prob/MC
            mi=mi/MC
            a1b=a1b/MC
            a2b=a2b/MC
            mi_closured=mi_closured/MC
            misquared=misquared/MC
            mi_closured_squared=mi_closured_squared/MC
            mean=mean/MC
            secmom=secmom/MC
            rate=mi[:-1]/timevec
            rate_closured=mi_closured[:-1]/timevec
            return a_g[:-1],b_g[:-1],mean[:-1],secmom[:-1],mi[:-1],misquared[:-1],rate,a1b[:-1],a2b[:-1],mi_closured[:-1],mi_closured_squared[:-1],rate_closured
        else:
            for r in results:
                ag,bg,a1bg,a2bg,miclog,miclosqg = r
                a_g += ag
                b_g += bg
                prob += probg
                a1b += a1bg
                a2b += a2bg
                mi_closured += miclog
                mi_closured_squared += miclosqg 
            a_g=a_g/MC
            b_g=b_g/MC
            a1b=a1b/MC
            a2b=a2b/MC
            mi_closured=mi_closured/MC
            mi_closured_squared=mi_closured_squared/MC
            mean=mean/MC
            rate_closured=mi_closured[:-1]/timevec
            return a_g[:-1],b_g[:-1],a1b[:-1],a2b[:-1],mi_closured[:-1],mi_closured_squared[:-1],rate_closured
    
    
    
    
    
    
    
    
    
    
    
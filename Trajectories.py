#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 09:40:13 2022

@author: moor
"""

import numpy as np
from functions import FF3Node,FF2Node,matrixAC,matrixC ,matrixB,dN,dN2D,mean2d
import random as rnd
from scipy.integrate import odeint
from exactsolutions import dglAC,dglAB
from functools import partial
from multiprocessing import Pool

def trajectoryAC(n,exact,iniconds,const,dim,laenge,destime):
    bvec=np.zeros(dim*dim)
    for i in range(dim*dim):
        bvec[i]=i%dim
    updates=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
    a0,b0,c0=iniconds
    if exact==True:
        Pt_mi=np.zeros((1,dim*dim+dim+1))
        dydt=np.zeros(dim*dim+dim+1)
        Pt_mi[0][a0]=1
        Pt_mi[0][dim+1+dim*a0+b0]=1
        a_g,b_g,c_g,mitot,b1_AC,b2_AC,b1_C,b2_C,a1_C,a2_C,ab_C,miclo=[np.zeros(laenge) for i in range(12)]
        dPAC1,dPAC2=[np.zeros(dim) for i in range(2)]
        dPC1,dPC2=[np.zeros(dim*dim) for i in range(2)]
    
        t=0
        tau=0
        Pt_ac=np.zeros((laenge,dim))
        Pt_c=np.zeros((laenge,dim*dim))
        a=a0
        b=b0
        c=c0
        a_g[0]=a0
        b_g[0]=b0
        c_g[0]=c0
        t_max=destime[-1]
        mAC=np.zeros((dim,dim))
        matrixAC(dim,a,const,mAC)
        mC=np.zeros((dim*dim,dim*dim))
        matrixC(dim,const,mC)
        j=0
        h=0
        sol=np.array([[b0, b0*b0, a0,b0,a0*a0,b0*b0,a0*b0,0]])
        propensities=[const[0], const[1]*a, const[2]*a, const[3]*b, const[4]*b, const[5]*c]
        while (t<t_max):
            reacsum=sum(propensities)
            time=-np.log(rnd.uniform(0.0,1.0))/reacsum     
            RV=rnd.uniform(0.0,1.0)*reacsum      
            index=1
            value=propensities[0]
            while value<RV:
                index+=1
                value+=propensities[index-1]  
            propensities=[const[0], const[1]*a, const[2]*a, const[3]*b, const[4]*b, const[5]*c]
            if h<len(destime):
                while destime[h]<(t+time): 
                    y0=[sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4],sol[-1,5],sol[-1,6],sol[-1,7]]
                    P0=Pt_mi[-1].copy()
                    timevec=np.linspace(tau,destime[h],5)
                    sol=odeint(FF3Node,y0,timevec, args=(a,const))
                    Pt_mi=odeint(dglAC,P0,timevec, args=(mAC,mC,a,dim,const,dPAC1,dPAC2,dPC1,dPC2,dydt,bvec))
                    tau=destime[h]
                    b1_AC[h]=sol[-1,0]
                    b2_AC[h]=sol[-1,1]
                    a1_C[h]=sol[-1,2]
                    a2_C[h]=sol[-1,4]
                    b1_C[h]=sol[-1,3]
                    b2_C[h]=sol[-1,5]
                    ab_C[h]=sol[-1,6]
                    miclo[h]=sol[-1,7]
                    Pt_mi[-1][:dim]=Pt_mi[-1][:dim].copy()/np.sum(Pt_mi[-1][:dim])
                    Pt_mi[-1][dim+1:]=Pt_mi[-1][dim+1:].copy()/np.sum(Pt_mi[-1][dim+1:])
                    Pt_ac[h]=Pt_mi[-1][:dim].copy()
                    Pt_c[h]=Pt_mi[-1][dim+1:].copy()
                    mitot[h]=Pt_mi[-1][dim]
                    a_g[h]=a
                    b_g[h]=b
                    h=h+1
                    j=j+1
                    # print(h)
                    if h>=len(destime)-1:
                        break
    
                y0=[sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4],sol[-1,5],sol[-1,6],sol[-1,7]]
                P0=Pt_mi[-1].copy()
    
                timevec=np.linspace(tau,t+time,5)
                sol=odeint(FF3Node,y0,timevec, args=(a,const))
                Pt_mi=odeint(dglAC,P0,timevec, args=(mAC,mC,a,dim,const,dPAC1,dPAC2,dPC1,dPC2,dydt,bvec))
                Pt_mi[-1][:dim]=Pt_mi[-1][:dim].copy()/np.sum(Pt_mi[-1][:dim])
                Pt_mi[-1][dim+1:]=Pt_mi[-1][dim+1:].copy()/np.sum(Pt_mi[-1][dim+1:])
                tau=t+time
            t=t+time
            if index==5:
                b1_AC[h]=(sol[-1,1]-sol[-1,0]*sol[-1,0])/sol[-1,0]+sol[-1,0]
                b2_AC[h]=sol[-1,1]+(2*sol[-1,1]*sol[-1,1]/sol[-1,0]-sol[-1,0]*sol[-1,1]*2)/sol[-1,0] #Gamma
                b1_C[h]=sol[-1,3]+(sol[-1,5]-sol[-1,3]*sol[-1,3])/sol[-1,3]
                b2_C[h]=sol[-1,5]+(2*(sol[-1,5]*sol[-1,5]/sol[-1,3])-sol[-1,3]*sol[-1,5]*2)/sol[-1,3] #Gamma
                a1_C[h]=sol[-1,2]+(sol[-1,6]-sol[-1,2]*sol[-1,3])/sol[-1,3]
                a2_C[h]=sol[-1,4]+(2*(sol[-1,4]*sol[-1,6]/sol[-1,2])-sol[-1,4]*sol[-1,3]*2)/sol[-1,3] #Gamma
                ab_C[h]=sol[-1,6]+(2*(sol[-1,5]*sol[-1,6]/sol[-1,3])-sol[-1,5]*sol[-1,2]-sol[-1,3]*sol[-1,6])/sol[-1,3] #Gamma
                if sol[-1,0]==0 and sol[-1,3]!=0:
                    miclo[h]=sol[-1,7]+(-np.log(const[4]*sol[-1,3]))
                if sol[-1,0]!=0 and sol[-1,3]==0:
                    miclo[h]=sol[-1,7]+(np.log(const[4]*sol[-1,0]))
                if sol[-1,0]==0 and sol[-1,3]==0:
                    miclo[h]=sol[-1,7]
                if sol[-1,0]!=0 and sol[-1,3]!=0:
                    miclo[h]=sol[-1,7]+(np.log(const[4]*sol[-1,0])-np.log(const[4]*sol[-1,3]))
                sol[-1,7]=miclo[h]
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
                    mitot[h]=Pt_mi[-1][dim]-np.log(const[4]*bc)
                if bc==0 and bac!=0:
                    mitot[h]=Pt_mi[-1][dim]+np.log(const[4]*bac)
                if bc==0 and bac==0:
                    mitot[h]=Pt_mi[-1][dim]
                if bc!=0 and bac!=0:
                    mitot[h]=Pt_mi[-1][dim]+(np.log(const[4]*bac)-np.log(const[4]*bc))
                tau=t
                Pt_ac[h]=Pt_ac[h].copy()/np.sum(Pt_ac[h])
                Pt_c[h]=Pt_c[h].copy()/np.sum(Pt_c[h])
                Pt_mi[-1][:dim]=Pt_ac[h].copy()
                Pt_mi[-1][dim+1:]=Pt_c[h].copy()
                Pt_mi[-1][dim]=mitot[h]
            a+=updates[index-1][0]
            b+=updates[index-1][1]
            c+=updates[index-1][2]
            matrixAC(dim,a,const,mAC)
            matrixC(dim,const,mC)
        return a_g,b_g,Pt_ac,Pt_c,mitot,mitot*mitot,b1_AC,b1_C,miclo,miclo*miclo #b1_AC,b2_AC,b1_C,b2_C,a1_C,a2_C,ab_C
    else:
        a_g,b_g,c_g,b1_AC,b2_AC,b1_C,b2_C,a1_C,a2_C,ab_C,miclo=[np.zeros(laenge) for i in range(11)]
    
        t=0
        tau=0
        a=a0
        b=b0
        c=c0
        a_g[0]=a0
        b_g[0]=b0
        c_g[0]=c0
        t_max=destime[-1]
        j=0
        h=0
        sol=np.array([[b0, b0*b0, a0,b0,a0*a0,b0*b0,a0*b0,0]])
        propensities=[const[0], const[1]*a, const[2]*a, const[3]*b, const[4]*b, const[5]*c]
        while (t<t_max):
            reacsum=sum(propensities)
            time=-np.log(rnd.uniform(0.0,1.0))/reacsum     
            RV=rnd.uniform(0.0,1.0)*reacsum      
            index=1
            value=propensities[0]
            while value<RV:
                index+=1
                value+=propensities[index-1]  
            propensities=[const[0], const[1]*a, const[2]*a, const[3]*b, const[4]*b, const[5]*c]
            if h<len(destime):
                while destime[h]<(t+time): 
                    y0=[sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4],sol[-1,5],sol[-1,6],sol[-1,7]]
                    timevec=np.linspace(tau,destime[h],5)
                    sol=odeint(FF3Node,y0,timevec, args=(a,const))
                    tau=destime[h]
                    b1_AC[h]=sol[-1,0]
                    b2_AC[h]=sol[-1,1]
                    a1_C[h]=sol[-1,2]
                    a2_C[h]=sol[-1,4]
                    b1_C[h]=sol[-1,3]
                    b2_C[h]=sol[-1,5]
                    ab_C[h]=sol[-1,6]
                    miclo[h]=sol[-1,7]
                    a_g[h]=a
                    b_g[h]=b
                    h=h+1
                    j=j+1
                    if h>=len(destime)-1:
                        break
    
                y0=[sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4],sol[-1,5],sol[-1,6],sol[-1,7]]
    
                timevec=np.linspace(tau,t+time,5)
                sol=odeint(FF3Node,y0,timevec, args=(a,const))
                tau=t+time
            t=t+time
            if index==5:
                b1_AC[h]=(sol[-1,1]-sol[-1,0]*sol[-1,0])/sol[-1,0]+sol[-1,0]
                b2_AC[h]=sol[-1,1]+(2*sol[-1,1]*sol[-1,1]/sol[-1,0]-sol[-1,0]*sol[-1,1]*2)/sol[-1,0] #Gamma
                b1_C[h]=sol[-1,3]+(sol[-1,5]-sol[-1,3]*sol[-1,3])/sol[-1,3]
                b2_C[h]=sol[-1,5]+(2*(sol[-1,5]*sol[-1,5]/sol[-1,3])-sol[-1,3]*sol[-1,5]*2)/sol[-1,3] #Gamma
                a1_C[h]=sol[-1,2]+(sol[-1,6]-sol[-1,2]*sol[-1,3])/sol[-1,3]
                a2_C[h]=sol[-1,4]+(2*(sol[-1,4]*sol[-1,6]/sol[-1,2])-sol[-1,4]*sol[-1,3]*2)/sol[-1,3] #Gamma
                ab_C[h]=sol[-1,6]+(2*(sol[-1,5]*sol[-1,6]/sol[-1,3])-sol[-1,5]*sol[-1,2]-sol[-1,3]*sol[-1,6])/sol[-1,3] #Gamma
                if sol[-1,0]==0 and sol[-1,3]!=0:
                    miclo[h]=sol[-1,7]+(-np.log(const[4]*sol[-1,3]))
                if sol[-1,0]!=0 and sol[-1,3]==0:
                    miclo[h]=sol[-1,7]+(np.log(const[4]*sol[-1,0]))
                if sol[-1,0]==0 and sol[-1,3]==0:
                    miclo[h]=sol[-1,7]
                if sol[-1,0]!=0 and sol[-1,3]!=0:
                    miclo[h]=sol[-1,7]+(np.log(const[4]*sol[-1,0])-np.log(const[4]*sol[-1,3]))
                sol[-1,7]=miclo[h]
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
        return a_g,b_g,b1_AC,b1_C,miclo,miclo*miclo #b1_AC,b2_AC,b1_C,b2_C,a1_C,a2_C,ab_C
    
def trajectoryAB(n,exact,iniconds,const,dim,laenge,destime):
    updates=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]]
    a0,b0=iniconds
    if exact==True:
        Pt_mi=np.zeros((1,dim+1))
        Pt_mi[0][a0]=1
        dydt=np.zeros(dim+1)
        dP=np.zeros(dim)
        a_g,b_g,mitot,a1_B,a2_B,miclo=[np.zeros(laenge) for i in range(6)]
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
        mB=matrixB(dim,const[0],const[1],const[2])
        j=0
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
                    y0=[sol[-1,0],sol[-1,1],sol[-1,2]]
                    P0=Pt_mi[-1].copy()
                    timevec=np.linspace(tau,destime[h],5)
                    sol=odeint(FF2Node,y0,timevec, args=(a,const))
                    Pt_mi=odeint(dglAB,P0,timevec, args=(mB,a,dim,const,dP,dydt))
                    tau=destime[h]
                    a1_B[h]=sol[-1,0]
                    a2_B[h]=sol[-1,1]
                    miclo[h]=sol[-1,2]
                    Pt_mi[-1][:-1]=Pt_mi[-1][:-1].copy()/np.sum(Pt_mi[-1][:-1])
                    Pt_tr[h]=Pt_mi[-1][:-1].copy()
                    mitot[h]=Pt_mi[-1][-1]
                    a_g[h]=a
                    b_g[h]=b
                    h=h+1
                    j=j+1
                    if h>=len(destime)-1:
                        break
                y0=[sol[-1,0],sol[-1,1],sol[-1,2]]
                P0=Pt_mi[-1].copy()
                timevec=np.linspace(tau,t+time,5)
                sol=odeint(FF2Node,y0,timevec, args=(a,const))
                Pt_mi=odeint(dglAB,P0,timevec, args=(mB,a,dim,const,dP,dydt))
                Pt_mi[-1][:-1]=Pt_mi[-1][:-1].copy()/np.sum(Pt_mi[-1][:-1])
                tau=t+time
            t=t+time
            if index==3:
                if a==0 and sol[-1,0]!=0:
                    miclo[h]=sol[-1,2]+(-np.log(const[2]*sol[-1,0]))
                if a!=0 and sol[-1,0]==0:
                    miclo[h]=sol[-1,2]+(np.log(const[2]*a))
                if a==0 and sol[-1,0]==0:
                    miclo[h]=sol[-1,2]
                if a!=0 and sol[-1,0]!=0:
                    miclo[h]=sol[-1,2]+(np.log(const[2]*a)-np.log(const[2]*sol[-1,0]))
                a1_B[h]=sol[-1,0]+(sol[-1,1]-sol[-1,0]*sol[-1,0])/sol[-1,0]
                a2_B[h]=sol[-1,1]+(2*(sol[-1,1]/(sol[-1,0]*sol[-1,0]))*(sol[-1,1]-sol[-1,0]*sol[-1,0]))
                Pt_tr[h]=Pt_mi[-1][:-1].copy()+dN(Pt_mi[-1][:-1].copy(),dim)
                if np.dot(Pt_mi[-1][:-1],[k for k in range(dim)])==0 and a!=0:
                    mitot[h]=Pt_mi[-1][-1]+np.log(const[2]*a)
                if a==0 and np.dot(Pt_mi[-1][:-1],[k for k in range(dim)])!=0:
                    mitot[h]=Pt_mi[-1][-1]-np.log(const[2]*(np.dot(Pt_mi[-1][:-1],[k for k in range(dim)])))
                if a==0 and np.dot(Pt_mi[-1][:-1],[k for k in range(dim)])==0:
                    mitot[h]=Pt_mi[-1][-1]
                if a!=0 and np.dot(Pt_mi[-1][:-1],[k for k in range(dim)])!=0:
                    mitot[h]=Pt_mi[-1][-1]+(np.log(const[2]*a)-np.log(const[2]*(np.dot(Pt_mi[-1][:-1],[k for k in range(dim)]))))
                tau=t
                Pt_tr[h]=Pt_tr[h].copy()/np.sum(Pt_tr[h])
                sol[-1,2]=miclo[h]
                sol[-1,0]=a1_B[h]
                sol[-1,1]=a2_B[h]
                Pt_mi[-1][:-1]=Pt_tr[h].copy()
                Pt_mi[-1][-1]=mitot[h]
            a+=updates[index-1][0]
            b+=updates[index-1][1]
        return a_g,b_g,Pt_tr,mitot,mitot*mitot,a1_B,a2_B,miclo,miclo*miclo
    else:
        a_g,b_g,mitot,a1_B,a2_B,miclo=[np.zeros(laenge) for i in range(6)]
        a1_B[0]=a0
        a2_B[0]=a0*a0
        t=0
        tau=0
        a=a0
        b=b0
        a_g[0]=a0
        b_g[0]=b0
        t_max=destime[-1]
        j=0
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
                    sol=odeint(FF2Node,y0,timevec, args=(a,const))
                    tau=destime[h]
                    a1_B[h]=sol[-1,0]
                    a2_B[h]=sol[-1,1]
                    miclo[h]=sol[-1,2]
                    a_g[h]=a
                    b_g[h]=b
                    h=h+1
                    j=j+1
                    if h>=len(destime)-1:
                        break
                timevec=np.linspace(tau,t+time,5)
                sol=odeint(FF2Node,y0,timevec, args=(a,const))
                tau=t+time
            t=t+time
            if index==3:
                if a==0 and sol[-1,0]!=0:
                    miclo[h]=sol[-1,2]+(-np.log(const[2]*sol[-1,0]))
                if a!=0 and sol[-1,0]==0:
                    miclo[h]=sol[-1,2]+(np.log(const[2]*a))
                if a==0 and sol[-1,0]==0:
                    miclo[h]=sol[-1,2]
                if a!=0 and sol[-1,0]!=0:
                    miclo[h]=sol[-1,2]+(np.log(const[2]*a)-np.log(const[2]*sol[-1,0]))
                a1_B[h]=sol[-1,0]+(sol[-1,1]-sol[-1,0]*sol[-1,0])/sol[-1,0]
                a2_B[h]=sol[-1,1]+(2*(sol[-1,1]/(sol[-1,0]*sol[-1,0]))*(sol[-1,1]-sol[-1,0]*sol[-1,0]))
                tau=t
                sol[-1,2]=miclo[h]
                sol[-1,0]=a1_B[h]
                sol[-1,1]=a2_B[h]
            a+=updates[index-1][0]
            b+=updates[index-1][1]
        return a_g,b_g,a1_B,a2_B,miclo,miclo*miclo
    
def parallelAC(core,MC,exact,iniconds,const,dim,laenge,timevec):
    a_g,b_g,mi,b1ac,b1c,miclo,misq,miclosq,meanac,meanc=[np.zeros(laenge) for i in range(10)]
    miclodata=[]
    miexdata=[]
    probc=np.zeros((laenge,dim*dim))
    probac=np.zeros((laenge,dim))
    with Pool(core) as pool:
        results = pool.map(partial(trajectoryAC,exact=exact,iniconds=iniconds,const=const,dim=dim,laenge=laenge,destime=timevec), [k for k in range(MC)])
        if exact==True:
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
                miclo += miclog
                miclodata.append(miclo)
                miexdata.append(mig)
                misq += misqg
                miclosq += miclosqg 
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
            miclo=miclo/MC
            misq=misq/MC
            miclosq=miclosq/MC
            meanac=meanac/MC
            meanc=meanc/MC
            rate=mi[:-1]/timevec
            rateclo=miclo[:-1]/timevec
            return a_g,b_g,meanac,meanc,mi,misq,rate,miexdata,b1ac,b1c,miclo,miclosq,rateclo,miclodata
        else:
            for r in results:
                ag,bg,b1acg,b1cg,miclog,miclosqg = r
                a_g += ag
                b_g += bg
                b1ac += b1acg
                b1c += b1cg
                miclo += miclog
                miclosq += miclosqg 
                miclodata.append(miclog)
            a_g=a_g/MC
            b_g=b_g/MC
            b1ac=b1ac/MC
            b1c=b1c/MC
            miclo=miclo/MC
            miclosq=miclosq/MC
            rateclo=miclo[:-1]/timevec
            return a_g,b_g,b1ac,b1c,miclo,miclosq,rateclo,miclodata  
        
def parallelAB(core,MC,exact,iniconds,const,dim,laenge,timevec):
    a_g,b_g,mi,a1b,a2b,miclo,misq,miclosq,mean,secmom=[np.zeros(laenge) for i in range(10)]
    prob=np.zeros((laenge,dim))
    with Pool(core) as pool:
        results = pool.map(partial(trajectoryAB,exact=exact,iniconds=iniconds,const=const,dim=dim,laenge=laenge,destime=timevec), [k for k in range(MC)])
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
                miclo += miclog
                misq += misqg
                miclosq += miclosqg 
            a_g=a_g/MC
            b_g=b_g/MC
            prob=prob/MC
            mi=mi/MC
            a1b=a1b/MC
            a2b=a2b/MC
            miclo=miclo/MC
            misq=misq/MC
            miclosq=miclosq/MC
            mean=mean/MC
            secmom=secmom/MC
            rate=mi[:-1]/timevec
            rateclo=miclo[:-1]/timevec
            return a_g[:-1],b_g[:-1],mean[:-1],secmom[:-1],mi[:-1],misq[:-1],rate,a1b[:-1],a2b[:-1],miclo[:-1],miclosq[:-1],rateclo
        else:
            for r in results:
                ag,bg,a1bg,a2bg,miclog,miclosqg = r
                a_g += ag
                b_g += bg
                prob += probg
                a1b += a1bg
                a2b += a2bg
                miclo += miclog
                miclosq += miclosqg 
            a_g=a_g/MC
            b_g=b_g/MC
            a1b=a1b/MC
            a2b=a2b/MC
            miclo=miclo/MC
            miclosq=miclosq/MC
            mean=mean/MC
            rateclo=miclo[:-1]/timevec
            return a_g[:-1],b_g[:-1],a1b[:-1],a2bv,miclo[:-1],miclosq[:-1],rateclo
    
    
    
    
    
    
    
    
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:27:39 2022

@author: moor
"""

import numpy as np

def evolveQuasiExactAC(y,t,AC,C,a,d,c,dPAC1,dPAC2,dPC1,dPC2,dydt,abvec):
    '''
    set of differential equations for the evaluation of the path mutual information of the three node network via the quasi-exact method
    '''
    PAC,PC,mi=y[:d],y[d+1:],y[d]
    np.dot(AC,PAC,out=dPAC1) 
    np.dot(c[4]*np.dot(PAC,[k for k in range(d)]),PAC,out=dPAC2)
    np.dot(C,PC,out=dPC1)
    np.dot(c[4]*np.dot(abvec,PC),PC,out=dPC2)
    dmi = -c[4]*(np.dot(PAC/np.sum(PAC),[k for k in range(d)])-np.dot(abvec,PC))
    dydt[:d]=dPAC1[:]+dPAC2[:]
    dydt[d]=dmi
    dydt[d+1:]=dPC1[:]+dPC2[:]
    return dydt

def evolveQuasiExactAB(y,t,A,a,d,c,dP,dydt):
    '''
    set of differential equations for the evaluation of the path mutual information of the two node network via the quasi-exact method
    '''
    P,mi=y[:-1],y[-1]
    np.dot(A,P,out=dP)
    dmi = -c[2]*(a-np.dot(P/np.sum(P),[k for k in range(d)]))
    dydt[:-1]=dP[:]
    dydt[-1]=dmi
    return dydt

def evolveQuasiExactBS(y,t,AC,C,A,d,c,dPAC1,dPAC2,dPC1,dPC2,dPA1,dPA2,dydt,abvec,bcvec):
    '''
    set of differential equations for the evaluation of the path mutual information of the three node network via the quasi-exact method
    '''
    PAC,PC,PA,mi=y[:d],y[d+1:d*d+d+1],y[d+1+d*d:],y[d]
    np.dot(AC,PAC,out=dPAC1) 
    np.dot(c[4]*np.dot(PAC,[k for k in range(d)]),PAC,out=dPAC2)
    np.dot(C,PC,out=dPC1)
    np.dot(c[4]*np.dot(abvec,PC),PC,out=dPC2)
    np.dot(A,PA,out=dPA1)
    np.dot(np.dot(bcvec,PA),PA,out=dPA2)
    dmi = -c[4]*(np.dot(PAC/np.sum(PAC),[k for k in range(d)])-np.dot(abvec,PC))-(c[0]-np.dot(bcvec,PA)) #normierung? 
    dydt[:d]=dPAC1[:]+dPAC2[:]
    dydt[d]=dmi
    dydt[d+1:d*d+d+1]=dPC1[:]+dPC2[:]
    dydt[d*d+d+1:]=dPA1[:]+dPA2[:]
    return dydt
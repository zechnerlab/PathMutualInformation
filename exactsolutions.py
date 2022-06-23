#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:27:39 2022

@author: moor
"""

import numpy as np

def dglAC(y,t,A,C,a,d,c,dPAC1,dPAC2,dPC1,dPC2,dydt,bvec):
    PAC,PC,mi=y[:d],y[d+1:],y[d]
    np.dot(A,PAC,out=dPAC1) 
    np.dot(c[4]*np.dot(PAC,[k for k in range(d)]),PAC,out=dPAC2)
    np.dot(C,PC,out=dPC1)
    np.dot(c[4]*np.dot(bvec,PC),PC,out=dPC2)
    dmi = -c[4]*(np.dot(PAC/np.sum(PAC),[k for k in range(d)])-np.dot(bvec,PC)) #mi mean2d(PC/np.sum(PC),d)
    dydt[:d]=dPAC1[:]+dPAC2[:]
    dydt[d]=dmi
    dydt[d+1:]=dPC1[:]+dPC2[:]
    return dydt

def dglAB(y,t,A,a,d,c,dP,dydt):
    P,mi=y[:-1],y[-1]
    np.dot(A,P,out=dP)
    dmi = -c[2]*(a-np.dot(P/np.sum(P),[k for k in range(d)])) #mi
    dydt[:-1]=dP[:]
    dydt[-1]=dmi
    return dydt
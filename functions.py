#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:25:22 2022

@author: moor
"""

import numpy as np

def FF3Node(y,t,a,constants):
    c_1,c_2,c_3,c_4,c_5,c_6=constants
    b1ac,b2ac,a1c,b1c,a2c,b2c,abc,mi=y
    dydt=[c_3*a-c_4*b1ac-c_5*b2ac+c_5*b1ac*b1ac, #b1ac
          2*c_3*a*b1ac+c_3*a-2*c_4*b2ac+c_4*b1ac-2*c_5*b2ac/b1ac*(b2ac-b1ac*b1ac), #b2ac
          c_1-c_2*a1c-c_5*abc+c_5*a1c*b1c, #a1c
          c_3*a1c-c_4*b1c-c_5*b2c+c_5*b1c*b1c, #b1c
          2*c_1*a1c+c_1-2*c_2*a2c+c_2*a1c-2*c_5*a2c*abc/a1c+c_5*a2c*b1c*2, #a2c
          2*c_3*abc+c_3*a1c-c_4*2*b2c+c_4*b1c-2*c_5*b2c*b2c/b1c+c_5*b1c*b2c*2, #b2c
          c_1*b1c-c_2*abc+c_3*a2c-c_4*abc-2*c_5*b2c*abc/b1c+c_5*b2c*a1c+c_5*b1c*abc, #abc
          -c_5*(b1ac-b1c) #mi 
          ]
    return dydt

def FF2Node(y,t,a,constants):
    a1,a2,mi=y
    c_1,c_2,c_3,c_4=constants
    dydt=[c_1-c_2*a1-c_3*(a2-a1*a1), #a1
           c_1+(2*c_1+c_2)*a1-2*c_2*a2-2*c_3*(a2/a1)*(a2-a1*a1), #a2
          -c_3*(a-a1) #mi
        ]
    return dydt

def matrixAC(d,a,c,mAC):
    for i in range(d):
        for j in range(d):
            if i==j:
                mAC[i][j]=-(c[2]*a+c[3]*i+c[4]*i)
            if j==(i+1):
                mAC[i][j]=c[3]*(i+1)
            if j==(i-1):
                mAC[i][j]=c[2]*a
    return 


def matrixC(d,c,mC):
    for a in range(d*d):
        for b in range(d*d):
            if b==(a+d): 
                mC[a][b]=(int(a/d)+1)*c[1]
            if a==(b+d):
                mC[a][b]=c[0]
            if b==(a+1) and (b % d)!=0:
                mC[a][b]=(b % d)*c[3]
            if b==(a-1) and (a % d)!=0:
                mC[a][b]=int(a/d)*c[2]
            if a==b:
                mC[a][b]=-(c[0]+c[1]*int(a/d)+c[2]*int(a/d)+c[3]*(b % d)+c[4]*(b % d))
    return

def matrixB(d,c1,c2,c3):
    ma=np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            if i==j:
                ma[i][j]=-(c1+c2*i+c3*i)
            if j==(i+1):
                ma[i][j]=c2*(i+1)
            if j==(i-1):
                ma[i][j]=c1
    return ma

def dN(P,d):
    jump=np.zeros(d)
    for i in range(1,d):
        jump[i]=i-1
    for i in range(len(P)):
        P[i]=P[i]*jump[i]
    return P

def dN2D(P,d):
    jump=np.zeros(d)
    for i in range(1,d):
        jump[i]=i-1
    if np.shape(P)!= (d,d):
        P=np.reshape(P,(d,d))
    for a in range(d):
        for b in range(d):
            P[a][b]=P[a][b]*jump[b]
    P=np.matrix.flatten(P)
    return P

def mean2d(P,d):
    if np.shape(P)!= (d,d):
        P=np.reshape(P,(d,d))
    mean=0
    for b in range(d):
        for a in range(d):
            mean+=b*P[a][b]
    return mean

def variance(firstmoment,secondmoment):
    var=np.zeros(len(firstmoment))
    for i in range(len(firstmoment)):
        var[i]=np.sqrt(secondmoment[i]-firstmoment[i]*firstmoment[i])
    return var


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 10:19:43 2022

@author: moor
"""

import numpy as np
from scipy.integrate import odeint

def evolveFF3_analytical(y,t,const):
    '''
    set of differential equations for the analytical solution of the path mutual information for the three node feed forward network
    '''
    am,bm,s_AC,s_AC_cross,s_AC_a,var_AC,s_C_a,s_cross,s_C,covar_C,var_C_a,var_C,mi=y
    c1,c2,c3,c4,c5,c6=const
    dydt=[ c1-c2*am, #am
          am*c3-bm*c4, #bm
          c3*am+c4*bm-c5*s_AC*s_AC/bm-2*c4*s_AC, #S_AC
          -(c2+c4)*s_AC_cross+c3*s_AC_a, #S_AC_co
          c1+c2*am-2*c2*s_AC_a, #S_AC_a
          2*c3*s_AC_cross-2*c4*var_AC+c5*s_AC*s_AC/bm, #VAR_AC
          c1-2*c2*s_C_a+c2*am-c5*s_cross*s_cross/bm, #S_C_a
          c3*s_C_a-(c2+c4)*s_cross-c5*s_cross*s_C/bm, #s_co
          2*c3*s_cross+c3*am-2*c4*s_C+c4*bm-c5*s_C*s_C/bm, #s_C
          c3*var_C_a-(c2+c4)*covar_C+c5*s_cross*s_C/bm, #covar_C
          -2*c2*var_C_a+c5*s_cross*s_cross/bm, #var_C_a
          2*c3*covar_C-2*c4*var_C+c5*s_C*s_C/bm, #var_C
          0.5*c5*(var_AC-var_C)/bm #MI
          ]
    return dydt

def evolveFF2_analytical(y,t,constants):
    '''
    set of differential equations for the analytical solution of the path mutual information for the two node feed forward network
    '''
    am,sm,mi=y
    c1,c2,c3,c4=constants
    dydt=[ c1-c2*am, #am
          c1+c2*am-2*c2*sm-c3*sm*sm/am, #sm
          0.5*c3*sm/am #MI
          ]
    return dydt

def calculateReacvel_3nodes(vel_list,const,iniconds,timevec,species):
    '''

    Parameters
    ----------
    vel_list : array containing the values of the relative reaction velocity
    const : list containing the reaction constants
    iniconds : list with three integer entries; initial conditions of the system
    timevec : array of float; time vector of the integration
    species : string; determines which reaction velocity is to be analysied, one can choose between 'A', 'B' and 'C'

    Returns
    -------
    rateV : array of float; the rate corresponding to the reaction velocity and the parameters in const 

    '''
    a0,b0,c0=iniconds
    if species=='A':
        k=0
        div=a0
    if species=='B':
        k=2
        div=b0/a0
    if species=='C':
        k=4
        div=c0/b0
    rateV=np.zeros(len(vel_list))
    y0=[a0,b0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(len(vel_list)):
        const[k]=vel_list[i]
        const[k+1]=vel_list[i]/div
        sol=odeint(evolveFF3_analytical,y0,timevec, args=(const,))
        rateV[i]=0.5*const[4]*(sol[-1,5]-sol[-1,11])/sol[-1,1]     
    return rateV

def calculateReacvel_2nodes(vel_list,const,iniconds,timevec,species):
    '''

    Parameters
    ----------
    vel_list : array containing the values of the relative reaction velocity
    const : list containing the reaction constants
    iniconds : list with two integer entries; initial conditions of the system
    timevec : array of float; time vector of the integration
    species : string; determines which reaction velocity is to be analysied, one can choose between 'A', 'B' and 'C'

    Returns
    -------
    rateV : array of float; the rate corresponding to the reaction velocity and the parameters in const 

    '''
    a0,b0=iniconds
    if species=='A':
        k=0
        div=a0
    if species=='B':
        k=2
        div=b0/a0
    rateV=np.zeros(len(vel_list))
    y0=[a0,0,0]
    for i in range(len(vel_list)):
        const[k]=vel_list[i]
        const[k+1]=vel_list[i]/div
        sol=odeint(evolveFF2_analytical,y0,timevec, args=(const,))
        rateV[i]=0.5*const[2]*sol[-1,1]/sol[-1,0]
    return rateV


def gaussrate3node(w,c):
    '''
    Integrand gaussian mutual information rate of the three node network in steady state analytically. Needs to be integrated numerically
    '''
    c1,c2,c3,c4,c5,c6=c
    return (-1/(4*np.pi)*np.log(1-c2*c3*c4*c5/(c2*c4*(c3*c5+c2*(c4+c5))+(c2**2+c4*(c4+c5))*w**2+w**4)))

def pmirate2node(c):
    '''
    Calculates the path mutual information rate of the two node network in steady state analytically.

    '''
    return -c[1]/2+1/2*np.sqrt(c[1]*(c[1]+2*c[2]))

def gaussrate2node(c):
    '''
    Calculates the gaussian mutual information rate of the two node network analytically.
    '''
    return -c[1]/2+1/2*np.sqrt(c[1]*(c[1]+c[2]))








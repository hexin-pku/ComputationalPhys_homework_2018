#!/usr/bin/python3
# Filename: diff.py

# note, such as the ux doesn't stand the value of u', but u'~ux, with some coefficients!

import numpy as np
from numba import jit

@jit(nopython=True)
def umu(u,i,j,stp): # u[i+1,j]-u[i-1,j]
    if(i>0 and i< stp):
        return (u[i+1,j]-u[i-1,j])
    else:
        return (u[1,j]-u[stp-1,j])

@jit(nopython=True)
def ux(u,i,j,stp): # u[i+1,j]-u[i-1,j]
    if(i>0 and i< stp):
        return (u[i+1,j]-u[i-1,j])
    else:
        return (u[1,j]-u[stp-1,j])
    
@jit(nopython=True)           
def u2mu2(u,i,j,stp): # u[i+1,j]**2-u[i-1,j]**2
    if(i>0 and i< stp):
        x=(u[i+1,j]+u[i-1,j])*(u[i+1,j]-u[i-1,j])
    else:
        x=(u[1,j]+u[stp-1,j])*(u[1,j]-u[stp-1,j])
    return x;

@jit(nopython=True)
def uux_1(u,i,j,stp): # (u[i+1,j]-u[i-1,j])*(u[i+1,j]+u[i,j]+u[i-1,j])
    if(i>0 and i< stp):
        x=(u[i+1,j]+u[i,j]+u[i-1,j])*(u[i+1,j]-u[i-1,j])
        #if(x>10**5):
        #    print('overflow,x ',x)
        #    exit()
    else:
        x=(u[1,j]+u[0,j]+u[stp-1,j])*(u[1,j]-u[stp-1,j])
    return x;

@jit(nopython=True)
def u_3(u,i,j,stp): # (u[i+1,j]-u[i-1,j])*(u[i+1,j]+u[i,j]+u[i-1,j])
    if(i>0 and i< stp):
        x=(u[i+1,j]+u[i,j]+u[i-1,j])
        if(x>10**5):
            print('overflow,x ',x)
            exit()
    else:
        x=(u[1,j]+u[0,j]+u[stp-1,j])
    return x;

@jit(nopython=True)
def uux_2(u,i,j,stp): # uux_2 == u2mu2
    if(i>0 and i< stp):
        x=(u[i+1,j]+u[i-1,j])*(u[i+1,j]-u[i-1,j])
    else:
        x=(u[1,j]+u[stp-1,j])*(u[1,j]-u[stp-1,j])
    return x;

@jit(nopython=True)
def uxxx_n2(u,i,j,stp): # note u''' ~ uxxx_n2/(2*dx^3)
    if(i>1 and i< stp-1):
        x = (u[i+2,j]-2*u[i+1,j]+2*u[i-1,j]-u[i-2,j])
    elif(i==1):
        x = (u[3,j]-2*u[2,j]+2*u[0,j]-u[stp-1,j])
    elif(i==0 or i==stp):
        x = (u[2,j]-2*u[1,j]+2*u[stp-1,j]-u[stp-2,j])
    elif(i==stp-1):
        x = (u[1,j]-2*u[0,j]+2*u[stp-2,j]-u[stp-3,j])
    else:
        print("error")
    return x;

@jit(nopython=True)
def uxxx_n3(u,i,j,stp): #  note u''' ~ uxxx_n3/(8*dx^3)
    if(i>2 and i< stp-2):
        x = (u[i+3,j]-3*u[i+1,j]+3*u[i-1,j]-u[i-3,j])
    elif(i==2):
        x = (u[5,j]-3*u[3,j]+3*u[1,j]-u[stp-1,j])
    elif(i==1):
        x = (u[4,j]-3*u[2,j]+3*u[0,j]-u[stp-2,j])
    elif(i==0 or i==stp):
        x = (u[3,j]-3*u[1,j]+3*u[stp-1,j]-u[stp-3,j])
    elif(i==stp-1):
        x = (u[2,j]-3*u[0,j]+3*u[stp-2,j]-u[stp-4,j])
    elif(i==stp-2):
        x = (u[1,j]-3*u[stp-1,j]+3*u[stp-3,j]-u[stp-5,j])
    else:
        print("error")
    return x;

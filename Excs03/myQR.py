#!/usr/bin/python3
# Filename: myQR.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def norm(v):
    return np.sqrt(np.dot(v,v))


def QR_householder(A,n):
    Qfull=np.eye(n)
    R=A.copy()
    for i in range(n-1):
        v=R[i:,i].copy()
        nm_x=norm(v)
        v[0]=v[0]-nm_x
        Q=np.eye(n)
        Q[i:,i:]=Q[i:,i:]-2*np.outer(v,v)/np.dot(v,v)
        # update the Qfull(total orthogonal matrix) and R
        Qfull=np.dot(Qfull,Q)
        R=np.dot(Q,R)
    return (Qfull,R)

def QR_givens(A,n):
    # it's a routine with default multiplication by using "numpy.dot" directly, 
    #    which might be slower, but actually faster than the below algorithm!
    Qfull=np.eye(n)
    R=A.copy()
    for i in range(n):
        for k in range(i+1,n):
            G=np.eye(n)
            if(abs(R[i,i])>abs(R[k,i])):
                t=R[k,i]/R[i,i]
                c=1/np.sqrt(1+t**2);s=c*t;
            else:
                t=R[i,i]/R[k,i]
                s=1/np.sqrt(1+t**2);c=s*t;
            G[i,i]=c;G[k,k]=c;
            G[i,k]=s;G[k,i]=-s;
            # update the Qfull(total orthogonal matrix) and R
            Qfull=np.dot(Qfull,G)
            R=np.dot(G,R)
    return (Qfull,R)


def QR_givens_my(A,n):# some annotations are removed, which can be found in solution (c)
    Q=np.eye(n)
    R=A.copy()
    for i in range(n):
        for k in range(i+1,n):
        # 1 judge, 2 divide, 2 times, 1 add, 1 sqrt
        # construct Givens matrix from i-th and k-th rows and columns
            if(abs(R[i,i])>abs(R[k,i])): 
                t=R[k,i]/R[i,i]
                c=1/np.sqrt(1+t*t);s=c*t;
            else:
                t=R[i,i]/R[k,i]
                s=1/np.sqrt(1+t*t);c=s*t;
            for j in range(n): # 8n multiply, 4n addtion/subtraction
                tmp1=Q[j,i]*c-Q[j,k]*s;
                tmp2=Q[j,i]*s+Q[j,k]*c;
                Q[j,i]=tmp1;
                Q[j,k]=tmp2;
                tmp1=R[i,j]*c+R[k,j]*s;
                tmp2=-R[i,j]*s+R[k,j]*c
                R[i,j]=tmp1; R[k,j]=tmp2;
    return (Q,R)





#!/usr/bin/python3
# Filename: myThomas.py
# solve for
#
# | a c 0 ... b |
# | b a c ... 0 |
# | 0 b a ... 0 | = B
# | . . . ... . |
# | c 0 0 ... 0 |

import numpy as np
def quasiThomas(Sa,Sb,Sc,B,b,c):
    n=len(Sa)
    X=np.zeros(n)
    #
    Sas=Sa[1:];Sbs=Sb[1:];Scs=Sc[1:];Bs=B[1:]
    u=Thomas(Sas,Sbs,Scs,Bs)
    tmp=np.zeros(n-1);tmp[0]=-Sb[0];tmp[-1]=-c;
    v=Thomas(Sas,Sbs,Scs,tmp)
    #
    X[0]=(B[0]-Sc[0]*u[0]-b*u[-1]) / (Sa[0]+Sc[0]*v[0]+b*v[-1])
    X[1:]=u+X[0]*v
    #
    return X

def Thomas(Sa,Sb,Sc,B):
# LU decomposition by Thomas
    n=len(B)
    a=np.zeros(n);b=np.zeros(n-1);# Sc not change! so needn't to make a copy!
    a[0]=Sa[0]
    
    for j in range(1,n):
        b[j-1]=Sb[j-1]/a[j-1]
        a[j]=Sa[j]-b[j-1]*Sc[j-1]
# inverse generation
    X=np.zeros(n)
    M=np.zeros(n)
    X[0]=B[0]
    for j in range(1,n):
        X[j]=B[j]-b[j-1]*X[j-1]
    M[n-1]=X[n-1]/a[n-1]
    for j in range(n-2,-1,-1):
        M[j]=(X[j]-Sc[j]*M[j+1])/a[j]
    return M

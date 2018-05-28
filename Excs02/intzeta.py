#!/usr/bin/python3
# Filename: intzeta.py
import numpy as np
import pandas as pd

def zeta00(q2,n2in):
    n2=13 # the default max n^2
    if(n2in>n2):
        n2=int(n2in)
    
    nsmp=1000 # number of sections of integral
    rng=4
    y=np.linspace(1,1+rng,2*nsmp+1) # integral from 1 to 5, rng=4
    rpd=pd.read_csv('pw.dat')
    pw=rpd.values.T[1]
    #such as pw=[1,6,12,8,6,24,24,0,12,30,24], pw is power of n^2 of each 3-D integer! 
                                                #see more detials in pw.py
    
    A=np.zeros(n2)
    B=np.zeros(5) # 5 terms are adequate!
    f=np.zeros((5,2*nsmp+1))
    
    if(q2%1==0):
        return np.inf
    
# solve the A_{n^2} term
    A[0]=np.exp(q2)/(-q2)
    for i in range(1,n2):
        A[i]=np.exp(q2-i)/(i-q2)
# solve B_{0}
    tmp=1
    for i in range(1,100):
        tmp=tmp*q2/i
        B[0]=B[0]+1/(i-0.5)*tmp
# solve the B_{n^2} term
    for i in range(1,5):
        f[i,]=np.exp(q2/(y*y)-np.pi**2*i*y*y)
        for j in range(nsmp):
            B[i]=B[i]+(f[i,2*j]+4*f[i,2*j+1]+f[i,2*j+2])/6
        B[i]=rng*B[i]/nsmp
    Z=1/np.sqrt(4*np.pi)*np.dot(A,pw[0:n2])-np.pi+np.pi/2*B[0]+np.sqrt(np.pi)*np.dot(B[1:],pw[1:5])
    return Z
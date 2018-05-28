#!/usr/bin/python3
# Filename: pw.py

import numpy as np
import pandas as pd
idx1=[0,1,3,6]
out=np.zeros(100)
for i in range(100):
    lst=[]
    for j in range(int(np.sqrt(i))+1):
        for k in range(j+1):
            if(i-j*j-k*k<0):break
            l=int(np.sqrt(i-j*j-k*k))
            if(l<=k and l*l+j*j+k*k==i):
                lst.append((j,k,l))
    for tm in lst:
        n1=1;n2=0;
        (j,k,l)=tm
        if(l>0):
            n2=3
        elif(k>0):
            n2=2
        elif(j>0):
            n2=1
        else:
            pass
        if(k!=l):
            n1=n1+1
        if(k!=j):
            n1=n1+1
        out[i]=out[i]+idx1[n1]*2**(n2)
ll=pd.DataFrame({'out':out})
ll.to_csv('pw.dat')
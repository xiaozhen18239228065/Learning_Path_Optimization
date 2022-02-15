import re
import time
import os
import sys
import random

dim=1000
ans=[[0 for i in range(dim)] for j in range(dim)]
def ran():
    k=random.randint(1,1000)
    if k>2:return 0
    return 1
for i in range(dim):
    ans[i][i]=1
    for j in range(i+1,dim):
        k=ran()
        ans[i][j]=k
        if k==1:
            ans[j][i]=0
        else:
            ans[j][i]=ran()
f=open("multi_matrix_input.txt","w",encoding="utf-8")
for i in range(dim):
    a=""
    for j in range(dim):
        a+=str(ans[i][j])+" "
    f.write(a+"\n")
f.close()

import re
import time
import os
import sys


data={}

#MAX=28260
MAX=10602

for i in range(MAX):
    cnt=0
    f=open("./group/group"+str(i),"r",encoding="utf-8")
    for line in f:
        line=line.strip()
        if line=="":continue
        cnt+=1
    cnt=str(cnt)
    if cnt in data:
        data[cnt]+=1
    else:
        data[cnt]=1
    f.close()
f=open("./freq.txt","w",encoding="utf-8")
for key in data:
    f.write(key+","+str(data[key])+"\n")
f.close()

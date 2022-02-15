import re
import time
import os
import sys

def readdata():
    global data
    f=open("./jjxs_result.txt","r",encoding="utf-8")
    for line in f:
        line=line.strip()
        if line=="":continue
        line=line.split('#')
        data[line[0]]=float(line[1])
    f.close()

data={}
readdata()

data=sorted(data.items(), key = lambda asd:asd[1],reverse=True)
f=open("./jjxs_sort_result.txt","w",encoding="utf-8")
ans=0
cnt=0
for item in data:
    ans+=item[1]
    cnt+=1
    f.write(item[0]+"#"+str(item[1])+"\n")
f.write(str(float(ans/cnt))+"\n")
f.close()
print(float(ans/cnt))

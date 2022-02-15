# this code is calculate the JuJiXiShu

import re
import time
import os
import sys

def readdata():
    global data
    f=open("./yourdata.txt","r",encoding="utf-8")
    for line in f:
        line=line.strip()
        if line=="":continue
        line=line.split('#')
        data[line[0]]=line[1:]
    f.close()

def Cmn(m):
    return m*(m-1)/2


def fun(key):
    global data
    if key not in data:return 0
    if len(data[key])<=1:return 0
    alllink=set()
    temp=set()
    alllink.update(data[key])
    temp.update(data[key])
    ans=0
    for key in alllink:
        if key not in data:continue
        temp.remove(key)
        ans+=len(temp & set(data[key]))
        temp.add(key)
    ans=ans/2
    return float(ans/Cmn(len(alllink)))


data={}
readdata()

out=open("jjxs_result.txt","w",encoding="utf-8")
for key in data:
    out.write(key+"#"+str(fun(key))+"\n")
out.close()

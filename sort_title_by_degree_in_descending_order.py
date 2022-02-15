import re
import time
import os
import sys


data={}

f=open("./yourdata_result.txt","r",encoding="utf-8")
for line in f:
    line=line.strip()
    if line=="":continue
    line=line.split('#')
    title=line[0]
    degree=int(line[1])+int(line[2])
    data[title]=degree
f.close()
data=sorted(data.items(), key=lambda e:e[1],reverse=True)
f=open("./yourdata_result_sorted.txt","w",encoding="utf-8")
for item in data:
    f.write(item[0]+"#"+str(item[1])+"\n")
f.close()

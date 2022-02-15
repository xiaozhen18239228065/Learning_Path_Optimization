import re
import time
import os
import sys


data={}

f=open("./yourdata.txt","r",encoding="utf-8")
for line in f:
    line=line.strip()
    if line=="":continue
    line=line.split('#')
    title=line[0]
    if title in data:
        data[title][0]+=len(line)-1
    else:
        data[title]=[len(line)-1,0]
    for word in line[1:]:
        if word in data:
            data[word][1]+=1
        else:
            data[word]=[0,1]
f.close()
f=open("./yourdata_result.txt","w",encoding="utf-8")
for key in data:
    f.write(key+"#"+str(data[key][0])+"#"+str(data[key][1])+"\n")
f.close()

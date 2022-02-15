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
    if title=="Psychometrics":
        g=open("./political_fiction_result.txt","w",encoding="utf-8")
        print(len(line)-1)
        for word in line[1:]:
            g.write(word+"\n")
        g.close()
        break
f.close()

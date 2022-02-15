import re
import time
import os
import sys


degree={}
count=0

print("Reading degree data...")

f=open("./yourdata_result_sorted.txt","r",encoding="utf-8")
for line in f:
    line=line.strip()
    if line=="":continue
    line=int(line.split('#')[1])
    count+=line
f.close()
print("Reading all data completed!")

f=open("./total_degree_result.txt","w",encoding="utf-8")
f.write("total degree: "+str(count))
f.close()

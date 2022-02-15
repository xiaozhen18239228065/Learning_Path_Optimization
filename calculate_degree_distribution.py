import re
import time
import os
import sys


degree={}
def fun(num):
    if num>=50000:return 0
    if num>=10000:return 1
    if num>=1000:return 2
    if num>=100:return 3
    if num>=50:return 4
    if num>=15:return 5
    if num>=5:return 6
    return 7
print("Reading degree data...")
ans=[0,0,0,0,0,0,0,0]
f=open("./yourdata_result_sorted.txt","r",encoding="utf-8")
for line in f:
    line=line.strip()
    if line=="":continue
    line=int(line.split('#')[1])
    ans[fun(line)]+=1
f.close()
print("Reading all data completed!")

f=open("./group_degree_result.txt","w",encoding="utf-8")
f.write(">=50000:"+str(ans[0])+"\n")
f.write(">=10000:"+str(ans[1])+"\n")
f.write(">=1000:"+str(ans[2])+"\n")
f.write(">=100:"+str(ans[3])+"\n")
f.write(">=50:"+str(ans[4])+"\n")
f.write(">=15:"+str(ans[5])+"\n")
f.write(">=5:"+str(ans[6])+"\n")
f.write("<10:"+str(ans[7])+"\n")
f.close()

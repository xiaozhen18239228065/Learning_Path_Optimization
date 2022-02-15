
import re
import time
import os
import sys

data1={}
data2={}
man=set()
woman=set()
ans={}
def cmp(key,man1,man2):
    global data2
    for item in data2[key]:
        if item[0]==man1:
            return True
        if item[0]==man2:
            return False

def readdata():
    global data1
    global data2
    global man
    global woman
    f=open("./1_17_score_rev.csv","r",encoding="utf-8")
    for line in f:
        line=line.strip()
        if line=="":continue
        line=line.split(',')
        man.add(line[0])
        if line[0] in data1:
            data1[line[0]].append((line[1],float(line[2])))
        else:
            data1[line[0]]=[(line[1],float(line[2]))]
        woman.add(line[1])
        if line[1] in data2:
            data2[line[1]].append((line[0],float(line[2])))
        else:
            data2[line[1]]=[(line[0],float(line[2]))]
    f.close()

def findscore(_man,_woman):
    global data1
    for item in data1[_man]:
        if item[0]==_woman:
            return item[1]
readdata()

for key in data1:
    data1[key].sort(key=lambda x:x[1],reverse=True)
for key in data2:
    data2[key].sort(key=lambda x:x[1],reverse=True)

while len(man)>0:
    key=man.pop()
    for item in data1[key]:
        if item[0] in ans:
            if cmp(item[0],ans[item[0]],key):
                continue
            else:
                man.add(ans[item[0]])
                ans[item[0]]=key
        else:
            ans[item[0]]=key
        break




f=open("gs_rev_result.txt","w",encoding="utf-8")
for key in ans:
    f.write(ans[key]+","+key+","+str(findscore(ans[key],key))+"\n")
f.close()

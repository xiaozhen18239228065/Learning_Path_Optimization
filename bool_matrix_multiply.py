import re
import time
import os
import sys


def inimatrix():
    global dim
    return [[0 for i in range(dim)] for j in range(dim)]

def trans():
    global newN
    global N
    global dim
    for i in range(dim):
        temp=set()
        for j in range(dim):
            if i in N[j]:
                temp.add(j)
        newN.append(temp)

def readP():
    global N
    global dim
    f=open("./multi_matrix_input.txt","r",encoding="utf-8")
    for line in f:
        line=line.strip()
        if line=="":continue
        line=line.split()
        temp=set()
        for i in range(len(line)):
            if line[i]=="1":
                temp.add(i)
        N.append(temp)
    f.close()
    dim=len(N)
def fun(x,y):
    if len(x & y)>0:return 1
    return 0

def multi():
    global newN
    global dim
    global N
    ans=[set() for i in range(dim)]
    for i in range(dim):
        for j in range(dim):
            if fun(N[i],newN[j])==1:
                ans[i].add(j)
    N=ans[:]
def out():
    global N
    global dim
    f=open("bool_matrix_multiply_result.txt","w",encoding="utf-8")
    for i in range(dim):
        ans=""
        for j in range(dim):
            if j in N[i]:
                ans+="1 "
            else:
                ans+="0 "
        f.write(ans+"\n")
    f.close()


N=[]
dim=0
readP()
newN=[]
trans()
k=10
for i in range(k-1):
    start_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print(i)
    multi()
    end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print(start_time+"---->"+end_time)
out()

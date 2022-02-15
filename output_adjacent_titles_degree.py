import re
import time
import os
import sys


data={}
degree={}
print("Reading wiki data...")
f=open("./yourdata.txt","r",encoding="utf-8")
for line in f:
    line=line.strip()
    if line=="":continue
    line=line.split('#')
    data[line[0]]=line[1:]
f.close()
print("Reading degree data...")
f=open("./yourdata_result.txt","r",encoding="utf-8")
for line in f:
    line=line.strip()
    if line=="":continue
    line=line.split('#')
    degree[line[0]]=int(line[2])
f.close()
print("Reading all data completed!")
while True:
    word=input("input a word:")
    if word not in data:
        print("The word is not exist in the dictionary!")
        continue
    ans={}
    for link in data[word]:
        if link in degree:
            ans[link]=degree[link]
        else:
            ans[link]=0
    ans=sorted(ans.items(), key=lambda e:e[1],reverse=True)
    f=open("./WQY/"+word+".txt","w",encoding="utf-8")
    for item in ans:
        f.write(item[0]+"#"+str(item[1])+"\n")
    f.close()
    print("Process completed! Please check answer at the file: ./WQY/"+word+".txt")

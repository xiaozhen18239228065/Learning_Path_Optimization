
import re
import time
import os
import sys

start_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
f=open("../output/part-r-00000","r",encoding="utf-8")
#f=open("../test.txt","r",encoding="utf-8")
data={}
for line in f:
    line=line.strip()
    if line=="":continue
    idx=line.find('#')
    data[line[0:idx]]=line[idx+1:]
f.close()
print("Read all data complete!")
end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print(start_time+"---->"+end_time)
start_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
i=0
while True:
    if len(data)==0:break
    for tkey in data:
        key=tkey
        break
    ans=set()
    ans.add(key)
    f=open("./group/group"+str(i),"w",encoding="utf-8")
    while True:
        if len(ans)==0:break
        title=ans.pop()
        if title in data:
            ans.update(data[title].split('#'))
            f.write(title+"\n")
            del data[title]
    f.close()
    i+=1
end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print(start_time+"---->"+end_time)

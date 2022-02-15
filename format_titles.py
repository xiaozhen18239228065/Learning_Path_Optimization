

import re
import time
import os
import sys

def read_data():
    global data
    f=open("../result.txt","r",encoding="utf-8")
    for line in f:
        line=line.strip()
        if line=="":continue
        idx=line.find('#')
        data[line[0:idx]]=line[idx+1:]
    f.close()

start_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
data={}
read_data()
end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print("Read all data completed!")
print(start_time+"---->"+end_time)


start_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
del_cnt=0
f=open("../result.txt","r",encoding="utf-8")
out=open("../finall.txt","w",encoding="utf-8")
for line in f:
    line=line.strip().split('#')
    l=len(line)
    if l<=0:continue
    ans=line[0]
    flag=False
    for i in range(1,l):
        if line[i] in data:
            flag=True
            ans+="#"+line[i]
        else:del_cnt+=1
    if flag:
        out.write(ans)
        out.write("\n")
out.close()
f.close()

print(str(del_cnt)+" words has been deleted!")
end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print(start_time+"---->"+end_time)

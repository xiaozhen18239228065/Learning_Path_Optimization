import re
import time
import os
import sys

def read_data():
    global data
    f=open("../finall.txt","r",encoding="utf-8")
    for line in f:
        line=line.strip()
        if line=="":continue
        idx=line.find('#')
        data[line[0:idx]]=line
    f.close()

start_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
data={}
read_data()
end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print("Read all data completed!")
print(start_time+"---->"+end_time)

f=open("../group0","r",encoding="utf-8")
out=open("./yourdata.txt","w",encoding="utf-8")
for line in f:
    line=line.strip()
    if line=="":continue
    if line in data:
        out.write(data[line])
        out.write("\n")
out.close()
f.close()

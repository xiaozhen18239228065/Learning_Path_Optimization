
import re
import time
import os
import sys


def read_data():
    global data
    f=open("./yourdata.txt","r",encoding="utf-8")
    for line in f:
        line=line.strip()
        if line=="":continue
        idx=line.find('#')
        data[line[0:idx].lower()]=line[idx+1:].lower().split('#')
    f.close()
ans=[]
def find_route(idx,start_word,end_word,deep):
    global data
    global ans
    if deep==21:return True
    if idx>deep:return False
    if start_word==end_word:return True
    if start_word not in data:return False
    for i in range(len(data[start_word])):
        ans[idx+1]=data[start_word][i]
        if find_route(idx+1,data[start_word][i],end_word,deep):
            return True
    return False






#start_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
#MAX=28203
#group_data=[set() for i in range(MAX)]
#read_group_data()
#print("Read group data complete!")
#end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
#print(start_time+"---->"+end_time)

start_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
data={}
read_data()
print("Read data complete!")
end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print(start_time+"---->"+end_time)
start_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
f=open("./ZBL_data_process_result.txt","r",encoding="utf-8")
out=open("./ZBL_result.txt","w",encoding="utf-8")
for line in f:
    line=line.strip()
    if line=="":continue
    line=line.split('#')
    if len(line)<3:continue
    start_word=line[0]
    end_word=line[1]
    max_deep=int(line[2])
    ans=["" for i in range(max_deep+10)]
    ans[0]=start_word
    find_route(0,start_word,end_word,max_deep)
    print(ans)
    out.write(start_word+"#"+end_word+"#["+"#".join(ans)+"]"+"\n")
out.close()
f.close()
end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print(start_time+"---->"+end_time)

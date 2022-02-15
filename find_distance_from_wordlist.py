
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
        data[line[0:idx].lower()]=line[idx+1:].lower()
    f.close()
def find_route(start_word,end_word):
    global data
    if start_word==end_word:return 0
    MAX_DEEP=20
    visit=set()
    visit.add(start_word)
    search_list=[(start_word,0)]
    i=0
    while True:
        l=len(search_list)
        if i>=l:
            return MAX_DEEP+1
        deep=search_list[i][1]
        if deep>=MAX_DEEP:
            return MAX_DEEP+1
        if search_list[i][0] in data:
            words=data[search_list[i][0]].split('#')
            for word in words:
                if word in visit:continue
                if word==end_word:
                    return deep+1
                visit.add(word)
                search_list.append((word,deep+1))
        i+=1





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
f=open("./ZBL.csv","r",encoding="utf-8")
out=open("./ZBL_result.csv","w",encoding="utf-8")
for line in f:
    line=line.strip()
    if line=="":continue
    line=line.split(',')
    start_word=line[0]
    end_word=line[1]
    out.write(start_word+","+end_word+","+str(find_route(start_word.lower(),end_word.lower()))+"\n")
out.close()
f.close()
end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print(start_time+"---->"+end_time)

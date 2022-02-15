
import re
import time
import os
import sys


def read_group_data():
    global group_data
    global MAX
    for i in range(MAX):
        f=open("./group/group"+str(i),"r",encoding="utf-8")
        for line in f:
            line=line.strip()
            if line=="":continue
            group_data[i].add(line)
        f.close()
def read_wiki_data():
    global wiki_data
    f=open("../finall.txt","r",encoding="utf-8")
    for line in f:
        line=line.strip()
        if line=="":continue
        idx=line.find('#')
        wiki_data[line[0:idx]]=line[idx+1:]
    f.close()
def find_set_id(word):
    global group_data
    global MAX
    for i in range(MAX):
        if word in group_data[i]:
            return i
    return -1
def find_route(start_word,end_word):
    global wiki_data
    MAX_DEEP=6
    visit=set()
    visit.add(start_word)
    search_list=[(start_word,0)]
    i=0
    while True:
        l=len(search_list)
        if i>=l:
            print(len(search_list))
            return MAX_DEEP+1
        deep=search_list[i][1]
        if deep>=MAX_DEEP:
            print(len(search_list))
            return MAX_DEEP+1
        if search_list[i][0] in wiki_data:
            words=wiki_data[search_list[i][0]].split('#')
            for word in words:
                if word in visit:continue
                if word==end_word:
                    print(len(search_list))
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
wiki_data={}
read_wiki_data()
print("Read wiki data complete!")
end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print(start_time+"---->"+end_time)

while True:
    start_word=input("start word:")
    end_word=input("end word:")
    start_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print(find_route(start_word,end_word))
    end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print(start_time+"---->"+end_time)

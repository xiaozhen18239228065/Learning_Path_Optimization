
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
    f=open("../raw.txt","r",encoding="utf-8")
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
def find_route(idx,start_word,end_word):
    global wiki_data
    global ans
    global one_ans
    global route
    if idx>=MAX_DEEP:return
    if start_word in one_ans:return
    if start_word not in wiki_data:return
    one_ans.add(start_word)
    route[idx]=start_word
    if start_word==end_word:
        ans+=[route[:idx+1]]
        one_ans.remove(start_word)
        return
    temp=set()
    temp.update(wiki_data[start_word].split('#'))
    for word in temp:
        find_route(idx+1,word,end_word)
    one_ans.remove(start_word)
def printans():
    global ans
    for one_ans in ans:
        print(one_ans)





start_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
MAX=10602
group_data=[set() for i in range(MAX)]
read_group_data()
print("Read group data complete!")
end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print(start_time+"---->"+end_time)

start_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
wiki_data={}
read_wiki_data()
print("Read wiki data complete!")
end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print(start_time+"---->"+end_time)


MAX_DEEP=4
while True:
    start_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    start_word=input("start word:")
    start_id=find_set_id(start_word)
    while start_id==-1:
        print("The start word you input is not exist!")
        start_word=input("start word:")
        start_id=find_set_id(start_word)
    end_word=input("end word:")
    end_id=find_set_id(end_word)
    while end_id==-1:
        print("The end word you input is not exist!")
        end_word=input("end word:")
        end_id=find_set_id(end_word)
    if start_id!=end_id:
        print("Unreachable!")
    ans=[]
    route=["" for i in range(MAX)]
    one_ans=set()
    find_route(0,start_word,end_word)
    printans()
    end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print(start_time+"---->"+end_time)

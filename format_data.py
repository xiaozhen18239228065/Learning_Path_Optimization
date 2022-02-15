

import re
import time
import os
import sys
def bigger(word):
    global trans_cnt
    if word=="":return ""
    ans=word[0]
    if word[0]>='a' and word[0]<='z':
        ans=ans.upper()
        trans_cnt+=1
    return ans+word[1:]



start_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
f=open("../raw.txt","r",encoding="utf-8")
out=open("../result.txt","w",encoding="utf-8")
del_cnt=0
trans_cnt=0
for line in f:
    line=line.strip()
    if line=="":continue
    line=line.split('#')
    ans=bigger(line[0])
    temp=set()
    temp.add(ans)
    l=len(line)
    flag=False
    for i in range(1,l):
        new_word=bigger(line[i])
        if new_word in temp:continue
        flag=True
        ans+="#"+new_word
        temp.add(new_word)
    if not flag:
        del_cnt+=1
        continue
    out.write(ans)
    out.write("\n")

out.close()
f.close()
print(str(del_cnt)+" words has been deleted!")
print(str(trans_cnt)+" words has been transfored!")
end_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print(start_time+"---->"+end_time)

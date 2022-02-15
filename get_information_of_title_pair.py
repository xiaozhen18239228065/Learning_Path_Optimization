
import re
import time
import os
import sys


pagerank={}
degree={}
print("Reading pagerank data...")
f=open("../job/pagerank/output/part-r-00000","r",encoding="utf-8")
for line in f:
    line=line.strip()
    if line=="":continue
    line=line.split('#')
    if len(line)<2:continue
    pagerank[line[0].lower()]=line[1]
f.close()
print("Reading degree data...")
f=open("./yourdata_result.txt","r",encoding="utf-8")
for line in f:
    line=line.strip()
    if line=="":continue
    line=line.split('#')
    if len(line)<3:continue
    degree[line[0].lower()]=line[1]+","+line[2]
f.close()




print("Processing...")
f=open("./ZBL.csv","r",encoding="utf-8")
out=open("./ZBL_degree_pagerank.csv","w",encoding="utf-8")
for line in f:
    line=line.strip()
    if line=="":continue
    line=line.split(',')
    if len(line)<2:continue
    ans=""
    ans+=line[0]
    if line[0].lower() in pagerank:
        ans+=","+pagerank[line[0].lower()]
    else:
        ans+=",None"
    if line[0].lower() in degree:
        ans+=","+degree[line[0].lower()]+","
    else:
        ans+=",None,"
    ans+=line[1]
    if line[1].lower() in pagerank:
        ans+=","+pagerank[line[1].lower()]
    else:
        ans+=",None"
    if line[1].lower() in degree:
        ans+=","+degree[line[1].lower()]
    else:
        ans+=",None"
    out.write(ans+"\n")
out.close()
f.close()

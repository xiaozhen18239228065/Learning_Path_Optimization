'''
Created on 2017-3-19

@author: diana
'''
import pickle
import codecs
import os
import simple_target_lp
import multi_target_lp
from gensim.models import word2vec
import random

model = word2vec.KeyedVectors.load(r"wiki.en.text.model")
tempdir = os.getcwd()+ r'/simple_result'
record = codecs.open(os.path.join(tempdir,'record.txt'),'w','utf-8')
recorddic = {}
view = codecs.open(os.path.join(tempdir,'view.txt'),'w','utf-8')

with open('testdic.pickle', 'rb') as f:
    testdic = pickle.load(f)

print 'simple target testing......'    
for path in testdic.keys():
    recorddic[path] = {}
    for tc in testdic[path]:
        simple_target_lp.record_slp(tc, path, model, record, recorddic, view)
record.close()
view.close()
with open(os.path.join(tempdir,'recorddic.pickle'),'wb') as f:
    pickle.dump(recorddic, f)
    
# multi

tempdir = os.getcwd()+ r'/multi_result'
if not os.path.exists(tempdir):
    os.makedirs(tempdir)
print 'multi targets testing......'
all_tcs = []
for path in testdic.keys():
    for tc in testdic[path]:
        all_tcs.append((tc,path))

for test in range(20):        
    tcfiledic = {}
    index = random.sample(range(len(all_tcs)), 5)
#    print index
    for i in index:
        tcfiledic[all_tcs[i][0]] = all_tcs[i][1]
#    print tcfiledic
    multi_target_lp.record_multi_lp(tcfiledic, model, test)
    
print 'finished!'    
    
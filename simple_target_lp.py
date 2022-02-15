'''
Created on 2017-3-17

@author: diana
'''
#import pickle
#import wikipedia
import codecs
import re
import nltk
import urllib
import os
import time
import sys
import string

k = 15
# load wiki from the file
#wikidic = {}
#with open('wikidic.pickle', 'rb') as wikidicpickle:
#    wikidic = pickle.load(wikidicpickle)

#load the model
#model = word2vec.KeyedVectors.load(r"wiki.en.text.model")

#def wiki_dic(wiki_filename):
#    global wikidic
#    i = 0
#    with codecs.open(wiki_filename, encoding="utf-8") as data:
#        for line in data:
#            term = line.split('#')
#            wikidic[term[0]] = term[1:]

def get_page(concept):
    """get searching results in Wikipedia"""
    concept = filter(lambda x: x in string.printable, concept)
    concept = re.sub('[^0-9a-zA-Z\s\'-]', ' ', concept)
    concept = concept.strip()
    url = "https://en.wikipedia.org/w/index.php?action=raw&title="+concept
    tag = True
    while tag:
        try:
            f = urllib.urlopen(url)
        except IOError, e:
            print 'sleeping......'
            time.sleep(10.0)
            continue
        else:
            tag = False
    s = f.read()
    f.close()
    return s

def get_outlinks(concept):
    """get document vector"""
    s1 = get_page(concept)
    del_regex = '[^0-9a-zA-Z\s\'-]'  # '[\]=\[{}\*()]'
    remain_regex = '\[\[([^\[\]]+)\]\]'
#     remain_regex = '\[\[.+?\]\]'
    
#    word_list1 = []
    results1 = re.findall(remain_regex, s1)
#    i = 0
    outlinks = []
    for result in results1:
        result = re.sub(del_regex, ' ', result)
        result = re.sub('(\s)+', ' ', result)
        result = result.strip()
        if result.upper().find('JPG') <> -1:
            continue
#        rl = nltk.word_tokenize(result)
#        result = [s.lower() for s in result]
#        if concept in result:
#            result.remove(concept)
        if concept.lower() <> result.lower():
            outlinks.append(result.lower())
#        i += 1
#        if i == k:
#            break
    return outlinks

def precs_wiki(tc, model):
    """extract candidate prerequisite concepts set from Wikipedia """
#    if wikidic.has_key(tc):
#        return wikidic[tc]
#    else: # wiki redirection
    outlinks = get_outlinks(tc)
    tc = tc.lower()
    
    if len(outlinks) == 0:
        return [],[]
#    print 'outlinks: ', outlinks
    similarity = {}
    for j in range(0, len(outlinks)):
        tcl = nltk.word_tokenize(tc)
        linksl = nltk.word_tokenize(outlinks[j])
        s = 0.0
        if len(tcl) == 0 or len(linksl) == 0:
            similarity[outlinks[j]] = 0
        else:
            for tc0 in tcl:
                for c in linksl: 
                    try:
                        s += model.similarity(tc0, c)
                    except KeyError:
                        continue
            similarity[outlinks[j]] = s/(len(tcl)*len(linksl))
    
#    print 'similarity ', similarity
    sdic = sorted(similarity.iteritems(), key=lambda d:d[1], reverse = True)
#    print 'sdic :', sdic
    i = 0
    precs = []
    while i < k and i < len(sdic):
        precs.append(sdic[i][0])
        i += 1
    return sdic, precs 
    
def precs_book(tc, filepath, model):
    tc = tc.lower()
    c_precs = []
    i = 0
    j = 0
    tag = 0
    with codecs.open(filepath, 'r', 'utf-8') as indexfile:
        for line in indexfile:
            line = line.lower()
            l = line.split(u',')
            l[0] = l[0].strip()
            if len(l) < 2 or l[0].isdigit():
                continue
            if l[0] <> tc and tag == 0:
                if len(c_precs) < k:
                    c_precs.append(l[0])
                else:
                    c_precs[i%k] = l[0]
                i += 1
            else:
                if tag == 0:
                    tag = 1
                if l[0] <> tc:
                    c_precs.append(l[0])
                    j += 1
                    if j == k:
                        break  
    if tag == 0:
        return [], []
#    print 'c_precs :', c_precs
    similarity = {}
    for j in range(0, len(c_precs)):
        
        c_precs_l = nltk.word_tokenize(c_precs[j])
        tcl = nltk.word_tokenize(tc)
        s = 0.0
        if len(tcl) == 0 or len(c_precs_l) == 0:
            similarity[c_precs[j]] = 0
        else:
            for tc0 in tcl:
                for c in c_precs_l:
                    try:
                        s += model.similarity(tc0, c)
                    except KeyError:
                        continue
            similarity[c_precs[j]] = s/(len(tcl)*len(c_precs_l))
        
    sdic = sorted(similarity.iteritems(), key=lambda d:d[1], reverse = True)
    i = 0
    precs = []
    while i < k and i < len(sdic):
        precs.append(sdic[i][0])
        i += 1
    return sdic, precs 

def gen_single_lp(tc, rootDir, model):
    sfromwiki, precsfromwiki = precs_wiki(tc, model)

#    for parent, dirNames, fileNames in os.walk(rootDir):
#        for filename in fileNames:
#            if filename == 'sorted_index.txt':
    sfrombook, precsfrombook = precs_book(tc, os.path.join(rootDir, 'sorted_index.txt'), model)

    i = 0
    j = 0
    lp = []
    len1 = len(sfromwiki)
    len2 = len(sfrombook)
    if len1 == 0:
        return precsfromwiki, precsfrombook, precsfrombook
    if len2 == 0:
        return precsfromwiki, precsfrombook, precsfromwiki
    tag = True
    while i < len1 and j < len2 and tag:
            if sfromwiki[i][1] > sfrombook[j][1]:
                lp.append(sfromwiki[i][0])
                i += 1
            else:
                lp.append(sfrombook[j][0])
                j += 1
            if len(lp) == k:
                tag = False 
    if tag:
        while i < len1:
            lp.append(sfromwiki[i][0])
            i += 1
            if len(lp) == k:
                break
        while j < len2:
            lp.append(sfrombook[j][0])
            j += 1
            if len(lp) == k:
                break
    return precsfromwiki, precsfrombook, lp

def show_slp(tc, rootDir, model):
    print '--------------------------------------------------------------'
    print 'start: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print 'target: ', tc
    print
    precsfromwiki, precsfrombook, lp = gen_single_lp(tc, rootDir, model)
    print 'precs_wiki: '
    for c in precsfromwiki:
        print c,'->',
    print tc
    print
    print 'precs_book: '
    for c in precsfrombook:
        print c,'->',
    print tc
    print
    print 'learning path: '
    for c in lp:
        print c,'->',
    print tc
    print 'end: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))     
    print '--------------------------------------------------------------'
    print

def record_slp(tc, rootDir, model, record, recorddic, view):
    
    record.write('--------------------------------------------------------------\n')
    record.write('start: '+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'\n')
    record.write('target: '+tc+'\n\n')
    record.write('book: '+rootDir+'\n\n')
    precsfromwiki, precsfrombook, lp = gen_single_lp(tc, rootDir, model)
    record.write('precs_wiki: \n')
    for c in precsfromwiki:
        record.write(c+'->')
    record.write(tc)
    record.write('\n\n')
    record.write('precs_book: \n')
    for c in precsfrombook:
        record.write(c+'->')
    record.write(tc+'\n\n')
    record.write('learning path: \n')
    for c in lp:
        record.write(c+'->')
    record.write(tc)
    record.write('\n')
    record.write('end: '+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'\n')
    record.write('--------------------------------------------------------------\n')
    record.write('\n')
    
    recorddic[rootDir][tc] = (rootDir, precsfromwiki+[tc], precsfrombook+[tc], lp+[tc])
    
    view.write('target: '+tc+'\n')
    view.write('book: '+rootDir+'\n')
    view.write('precs_wiki: ')
    for c in precsfromwiki:
        view.write(c+'->')
    view.write(tc+'\n')
    view.write('precs_book: ')
    for c in precsfrombook:
        view.write(c+'->')
    view.write(tc+'\n')
    view.write('learning path: ')
    for c in lp:
        view.write(c+'->')
    view.write(tc+'\n')
    view.write('\n')
#    view.write(tc+'\n'+str(precsfromwiki+[tc])+'\n'+str(precsfrombook+[tc])+'\n'+str(lp+[tc])+'\n')
    
if __name__ == '__main__':
    tcs = ['IP address', 'Host']
    for tc in tcs:
        rootDir = r'/home/zhangfei/DIANA/similarity/books/' + sys.argv[1] # refer root dir
        show_slp(tc, rootDir)
        

'''
Created on 2016-12-15

@author: diana
'''

import simple_target_lp
import multi_target_lp
from gensim.models import word2vec
import codecs

model = word2vec.KeyedVectors.load(r"wiki.en.text.model")

def menu():
    print '''-----------------menu-----------------
    1. single target concept
    2. multi target concepts
    3. exit
--------------------------------------'''
    choice = raw_input('Please input your choice(1/2): ')
    if choice == '1':
        bookname = raw_input("Please input the book's name: ")
        tc = raw_input('Please input your target concept: ')
        rootDir = r'/home/zhangfei/DIANA/similarity/books/' + bookname
        simple_target_lp.show_slp(tc, rootDir, model)
        record = codecs.open('record.txt','a','utf-8')
        recorddic = {}
        view = codecs.open('view.txt','a','utf-8')
        recorddic[rootDir] = {}
        simple_target_lp.record_slp(tc, rootDir, model, record, recorddic, view)
        record.close()
        view.close()
        menu()
        
    elif choice == '2':
        tcfiledic = {}
        n = raw_input('Please input the number of the books you want to learn: ')
        print "Please input the book and the corresponding target concepts (separated by '#'): "
        for i in range(1, int(n)+1):
            tcs_str = raw_input(str(i)+': ')
            tcs = tcs_str.split('#')
            for tc in tcs[1:]:
                tcfiledic[tc] = tcs[0]
        multi_target_lp.show_mlp(tcfiledic, model)
#        multi_target_lp.record_multi_lp(tcfiledic, model)
        menu()
        
    elif choice == '3':
        e = raw_input('Do you really want to exit?(Y/N) ')
        if e == 'Y' or e == 'y':
            print 'Thanks for using!'
            exit(0)
        else:
            menu()
    else:
        print 'The choice you selected not exists!'
        menu()
                                           
if __name__ == '__main__':
    menu()
#    print '\nbuild_graph:'
#    print 'start: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
#    build_graph()
#    print 'end: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
##    menu()
#    ends = ['queue', 'list', 'array', 'value', 'stack', 'tree']
#    multilearning(ends)
#    graphOpe.display_graph(graph)
#    nodes = set(nodes)
                
    
                

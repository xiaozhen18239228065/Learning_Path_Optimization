'''
Created on 2017-3-19

@author: diana
'''
import graphOpe
import simple_target_lp
import matplotlib  
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import networkx as nx
import time
import os

def gen_help_multibranch(msts):
    branches = []
    pathlen = 0
    i = 0
    while i < len(msts[0]):
        pathlen += msts[0][i][2]
        branch = [msts[0][i][0], msts[0][i][1]]
        if i+1 >= len(msts[0]):
            branches.append(branch)
            break
        for j in range(i+1, len(msts[0])):
            if msts[0][j][0] == branch[-1]:
                branch.append(msts[0][j][1])
                pathlen += msts[0][j][2]
            else:
                branches.append(branch)
                i = j
                break
        if i <> j:
            branches.append(branch)
            i = j + 1
    return pathlen, branches

def gen_inter_tcs_lp(tcs):
    graph = graphOpe.build_graph(tcs)
#    print 'graph: ', graph
    msts = graphOpe.prim(graph)
    if len(msts) == 0:
        return [], 0, []
    pathlen, branches = gen_help_multibranch(msts)
#    print 'msts', msts
    return msts[0], pathlen, branches 

def gen_inter_tcs_lp2(tcs):
    graph = graphOpe.build_graph(tcs)
#    print 'graph: ', graph
    msts = graphOpe.prim(graph)
    if len(msts) == 0:
        return []
#    print 'msts', msts
    return msts[0]
    
def gen_multi_lp(tcfiledic, model):
    tcs = tcfiledic.keys()
    s_lps = {}
    for tc in tcs:
        rootDir = r'/home/zhangfei/DIANA/similarity/books/' + tcfiledic[tc] # refer root dir
        precsfromwiki, precsfrombook, s_lp = simple_target_lp.gen_single_lp(tc, rootDir, model)
        s_lps[tc] = (precsfromwiki, precsfrombook, s_lp)
    m_lp_edges, m_lp_len, m_lp_branches = gen_inter_tcs_lp(tcs)
    return s_lps, m_lp_edges, m_lp_len, m_lp_branches

def show_mlp(tcfiledic, model):
    tcs = tcfiledic.keys()
    print '--------------------------------------------------------------'
    print 'start: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print 'targets: ', tcs
    print
    s_lps, m_lp_edges, m_lp_len, m_lp_branches = gen_multi_lp(tcfiledic, model)
    print 'pathlen: ', m_lp_len
    print
    print 'learning path: '
    for j in range(len(m_lp_branches)):
        print 'branch ', j+1
        for i in range(len(m_lp_branches[j])-1):
            print '[',
            for c in s_lps[m_lp_branches[j][i]][2]:
                print c, ',',
            print m_lp_branches[j][i], ']', "->"
        print '[',
        for c in s_lps[m_lp_branches[j][-1]][2]:
            print c, ',',
        print m_lp_branches[j][-1], ']'
    print 'end: ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print '--------------------------------------------------------------'
    print
    G = nx.DiGraph()
    G.add_weighted_edges_from(m_lp_edges)
    pos=nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    plt.savefig(os.path.join(os.getcwd()+ r'/multi_result', str(tcs)+".png"))
#    G = nx.DiGraph()
#    lablesdic = {}
#    for node in tcs:
#        pre_path = s_lps[node][2]
#        pre_path.append(node)
#        G.add_node()
#        lablesdic[node] = str(pre_path)
#        for i in range(len(pre_path)-1):
#            G.add_edge(pre_path[i], pre_path[i+1])
#    for edge in m_lp_edges:
#        G.add_edge(edge[0], edge[1])
#    nx.draw_networkx(G)
#    plt.savefig("lp.png")        

def record_multi_lp(tcfiledic, model, test):
    tcs = tcfiledic.keys()
    m_lp_edges = gen_inter_tcs_lp2(tcs)
    if m_lp_edges == [] or m_lp_edges == [[]]:
        return
    G = nx.DiGraph()
    G.add_weighted_edges_from(m_lp_edges)
    pos=nx.spring_layout(G)
#    for edge in m_lp_edges:
#        G.add_edge(edge[0], edge[1])
#    nx.draw_networkx(G,pos)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    plt.savefig(os.path.join(os.getcwd()+ r'/multi_result', str(test)+".png"))
        
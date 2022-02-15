'''
Created on 2016-12-15

@author: diana
'''

#import word_similarity
from collections import defaultdict  
from heapq import heapify, heappop, heappush
import concept_steps

def dijkstra_raw(graph, from_node, to_node):  
    q, seen = [(0,from_node,())], set()  
    while q:  
        (cost,v1,path) = heappop(q)  
        if v1 not in seen:  
            seen.add(v1)  
            path = (v1, path)  
            if v1 == to_node:  
                return cost,path
            if graph.has_key(v1):
                for v2,c in graph[v1].items():  
                    if v2 not in seen:  
                        heappush(q, (cost+c, v2, path))  
    return float("inf"),[]  
  
def dijkstra(graph, from_node, to_node):  
    len_shortest_path = -1  
    ret_path = []  
    length,path_queue = dijkstra_raw(graph, from_node, to_node)  
    if len(path_queue) > 0:  
        len_shortest_path = length      ## 1. Get the length firstly;  
        ## 2. Decompose the path_queue, to get the passing nodes in the shortest path.  
        left = path_queue[0]  
        ret_path.append(left)       ## 2.1 Record the destination node firstly;  
        right = path_queue[1]  
        while len(right) > 0:  
            left = right[0]  
            ret_path.append(left)   ## 2.2 Record other nodes, till the source-node.  
            right = right[1]  
        ret_path.reverse()  ## 3. Reverse the list finally, to make it be normal sequence.  
    return len_shortest_path,ret_path

def prim(graph):  
    conn = defaultdict(list)
    for n1 in graph.keys():
        for n2, c in graph[n1].items():  
            conn[n1].append((c, n1, n2))  
    nodes = graph.keys()
    msts = []
    while nodes:   
        mst = []
        startnode = nodes[0]
        used = set()  
        used.add(startnode)  
        usable_edges = conn[startnode][:]  
        heapify(usable_edges)  
      
        while usable_edges:  
            cost, n1, n2 = heappop(usable_edges)  
            if n2 not in used:  
                used.add(n2)  
                mst.append((n1, n2, cost))
                if n1 in nodes:
                    nodes.remove(n1)
                if n2 in nodes:
                    nodes.remove(n2)  
       
                for e in conn[n2]:  
                    if e[2] not in used:  
                        heappush(usable_edges, e)
        msts.append(mst)  
    return msts
            
def build_graph(tcs):
    graph = {}
    for c0 in tcs:
        for c1 in tcs:
            if c0 <> c1:
                if graph.has_key(c0):
                    graph[c0][c1] = concept_steps.caculate_steps(c0, c1)
                else:
                    graph[c0] = {c1: concept_steps.caculate_steps(c0, c1)}
    return graph
        
#def display_graph(graph):
#    i = 0
#    for start in graph.keys():
#        for end in graph[start].keys():
#            print start, "->", end, ": ", graph[start][end], "     ",
#            i = i + 1
#            if i == 5:
#                print
#                i = 0
#    print

if __name__ == '__main__':
#    prs = []
#    relevancefile = open('relevance.txt')
#    for line in relevancefile:
#        l0 = line.split(': ')
#        l1 = l0[1].split('(')
#        prs.append(float(l1[0]))
    graph = {
#        'B': {'A': [1], 'D': [1], 'G': [2]},
#        'A': {'B': [5], 'D': [3], 'E': [12], 'F':[5]},
#        'D': {'B': [2]},
        'N': {'M': [3]},
        'M': {'N': [1]}}
#        'D': {'B': 1, 'G': 1, 'E': 1, 'A': 3},
#        'G': {'B': 2, 'D': 1, 'C': 2},
#        'C': {'G': 2, 'E': 1, 'F': 16},
#        'E': {'A': 12, 'D': 1, 'C': 1, 'F': 2},
#        'F': {'A': 5, 'E': 2, 'C': 16}}
#    print find_shortest_path(graph, ['A',0], 'C')  
    nodes = ('A', 'B', 'C', 'D', 'E', 'F', 'G')
#    paths = [['A','B', 'C', 'D', 'E'], ['C','A',  'D', 'E','B'],['B', 'C', 'D','A','E'],['A', 'C', 'B','D', 'E'] ]
#    subgraphs = []
#    for i in range(1, 5):
#        if i == 1:
#                pr = prs[0]
#        elif i == 4:
#            pr = prs[i-2]
#        else:
#            pr = (prs[i-2] + prs[i-1])*0.5
#        
#        subgraph = build_subgraph(paths[i-1], pr)
#        subgraphs.append(subgraph)
#        
#    graph = subgraphs[0]        
#    for i in range(1,len(subgraphs)):
#        union_subgraph(graph, subgraphs[i])
#    
#    display_graph(graph)
    
#    length,shortest_path = dijkstra(graph, 'B', 'E')
#    print find_shortest_path(nodes, graph, 'A')
    print prim(graph)
    
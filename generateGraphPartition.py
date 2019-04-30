#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:57:11 2019

@author: g18quint
"""

import networkx as nx
import metis
#from cStringIO import StringIO

"""
# Set up graph structure
G = nx.Graph()
G.add_edges_from([ (0,1), (0,2), (0,3), (1, 2), (3, 4) ])

# Add node weights to graph
for i, value in enumerate([1,3,2,4,3]):
    G.node[i]['node_value'] = value

# tell METIS which node attribute to use for 
G.graph['node_weight_attr'] = 'node_value' 

# Get at MOST two partitions from METIS
(cut, parts) = metis.part_graph(G, 2) 
# parts == [0, 0, 0, 1, 1]

# Assuming you have PyDot installed, produce a DOT description of the graph:
colors = ['red', 'blue']
for i, part in enumerate(parts):
    G.node[i]['color'] = colors[part]
nx.nx_pydot.write_dot(G, 'example.dot')
"""


class GenerateGraphPartition():
    
    def createEdgesGraph():
        print("hola")
        graph = nx.read_gexf("mediumLinkedin.gexf")
        print("hola")
        print(graph.edges())
        print(graph.nodes())
        (cut, parts) = metis.part_graph(graph, 2)

        print(parts)
"""
    def plotGraph(pydot_graph):
        

        
        # convert from networkx -> pydot
        #pydot_graph = G.to_pydot()
        
        # render pydot by calling dot, no file saved to disk
        png_str = pydot_graph.create_png(prog='dot')
        
        # treat the dot output string as an image file
        sio = StringIO()
        sio.write(png_str)
        sio.seek(0)
        img = mpimg.imread(sio)
        
        # plot the image
        imgplot = plt.imshow(img, aspect='equal')
        plt.show(block=False)
        

 
    
    def createEdgesGraph(self):
        predicted_values={}
        for n in graph:
            nbrs_attr_values=[]
            for nbr in graph.neighbors(n):
                if nbr in attr: #si vecino tiene empleo cargado
                    for val in attr[nbr]:
                        nbrs_attr_values.append(val)
            predicted_values[n]=[]
            if nbrs_attr_values: # non empty list
                # count the number of occurrence each value and returns a dict
                cpt=Counter(nbrs_attr_values)
                # take the most represented attribute value among neighbors
                a,nb_occurrence=max(cpt.items(), key=lambda t: t[1])
                predicted_values[n].append(a)
        return predicted_values
    """
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:57:11 2019

@author: g18quint
"""

import networkx as nx
import metis
import matplotlib.pyplot as plt
from matplotlib import pylab
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
    
    def createEdgesGraph(self):
        print("hola")
        graph = nx.read_gexf("mediumLinkedin.gexf")
        print("hola")
        print(graph.edges())
        print(graph.nodes())
        (cut, parts) = metis.part_graph(graph, 40)



        print(parts)
        self.draw_graph(graph, parts)

    def draw_graph(self, g, parts, node_attribute=None, list_of_values_of_attributes=None):
        """
        Draw the graph g.

        Parameters
        ----------
        g : graph
           A networkx graph
        node_attribute : string
           The name of the node attribute used to assign colors to the drawing
        list_of_values_of_attributes : list
            A list of all the potential values of node_attribute to assign one color
            per value.
        """
        # initialze Figure
        plt.figure(num=None, figsize=(80, 80), dpi=80)
        plt.axis('off')
        fig = plt.figure(1)

        pos = nx.spring_layout(g, iterations=100)

        ### beta
        values = []
        i = 0
        for node in g.nodes():
            #if parts[i] == 1:
                    # we arbitrarily take the first value
                   # values.append(2)
            #else:
            values.append(parts[i])
            i += 1

        nx.draw_networkx_nodes(g, pos, cmap=plt.get_cmap('jet'), node_color=values)
        ####
        """
        if node_attribute and list_of_values_of_attributes:
            # To associate colors to nodes according to an attribute, here college
            # build a color_map, one for each college
            color_map = {}
            i = 0.0
            for s in list_of_values_of_attributes:
                color_map[s] = i
                i += 1 / len(list_of_values_of_attributes)
            color_map[None] = 1  # for nodes without values for the attribute node_attribute

            # The values supplied to node_color should be in the same order as the nodes
            # listed in G.nodes(). We take an arbitrary mapping of values color_map and
            # generate the values list in the correct order
            # values = [color_map[G.node[node].get(node_attribute)] for node in G.nodes()] # for attributes encoded in the graph
            values = []
            for node in g.nodes():
                if node in node_attribute:
                    if node_attribute[node]:
                        # we arbitrarily take the first value
                        values.append(color_map[node_attribute[node][0]])
                else:
                    values.append(1)

            nx.draw_networkx_nodes(g, pos, cmap=plt.get_cmap('jet'), node_color=values)
        else:
            nx.draw_networkx_nodes(g, pos)
        """

        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos)

        cut = 1.00
        #xmax = cut * max(xx for xx, yy in pos.values())
        #ymax = cut * max(yy for xx, yy in pos.values())
        #plt.xlim(0, xmax)
        #plt.ylim(0, ymax)
        plt.show()
        plt.savefig('foo.png')
        pylab.close()
        del fig
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
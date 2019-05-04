#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import metis
import matplotlib.pyplot as plt
from collections import Counter


class GenerateGraphPartition:
    def __init__(self, empty_nodes):
        self.emptyNodes = empty_nodes
        self.nodesclusters = {}

    def createEdgesGraph(self, college):
        
        """
        print(graph.edges())
        print(graph.nodes())
        (cut, parts) = metis.part_graph(graph, 40)
        print("hola")
        #(cut2, parts2) = metis.partition(graph, 2)
        metisGraph = metis.networkx_to_metis(graph)
        print(metisGraph)
        graph['U27476']['U27661']['weight']=1
        print(graph['U27476']['U27661'])
        graph.graph['edge_weight_attr'] = 'weight'
        metisGraph = metis.networkx_to_metis(graph)
        (cutW, partsW) = metis.part_graph(metisGraph, 10)
        print("paso esto")
        print(graph)
        print(metisGraph)
        """
        graph = nx.read_gexf("mediumLinkedin.gexf")
        weightedGraph = self.setEdgesWeights(graph, college)
        weightedGraph.graph['edge_weight_attr'] = 'weight'
        metisWeightedGraph = metis.networkx_to_metis(weightedGraph)
        print("Llego hasta aca")
        
        #self.draw_graph(metisWeightedGraph,parts)
        
        (cutW, partsW) = metis.part_graph(metisWeightedGraph, 10)

        self.draw_graph(weightedGraph, partsW)


        #self.setNodesClustersDictionary(weightedGraph, parts)

        #comento el draw para que corra mas rapido
        #self.draw_graph(graph, parts)

        self.getSubgraphByClusterID(graph, 25)

        print(cut)
        print(parts)

    def setEdgesWeights(self, graph, attrs):
        print(attrs)
        attr=attrs
        for node in graph.nodes():
            
            for nbr in graph.neighbors(node):
                edgeWeight = 0
                if (node in attr) and (nbr in attr):
                   
                    for valNbr in attr[nbr]:
                        
                        for valNode in attr[node]:
                           
                            if valNode == valNbr:
                                edgeWeight =+ 1
                graph[node][nbr]['weight']=edgeWeight
               
        return graph

    def setNodesClustersDictionary(self, graph, parts):
        i = 0
        for node in graph.nodes():
            # Set dictionary of clusters
            self.nodesclusters[node] = parts[i]
            i += 1

        print("Dictionary:", self.nodesclusters)

    def getSubgraphByClusterID(self, graph, idCluster):
        subgraph = nx.Graph()

        for node in graph:
            if node in self.nodesclusters:
                if self.nodesclusters[node] == idCluster:
                    subgraph.add_node(node)

        return subgraph

    def pridictAttributesByCluster(self, graph, emptyNodes, attr):
        (cut, parts) = metis.part_graph(graph, 10, recursive = True)
        print(cut)
        self.setNodesClustersDictionary(graph, parts)

        predicted_values = {}
        for node in emptyNodes:
            nbrs_attr_values = []
            clusterID = self.nodesclusters[node]
            for nbr in self.getSubgraphByClusterID(graph, clusterID):
                if nbr in attr:
                    for val in attr[nbr]:
                        nbrs_attr_values.append(val)
            predicted_values[node] = []
            if nbrs_attr_values:  # non empty list
                # count the number of occurrence each value and returns a dict
                cpt = Counter(nbrs_attr_values)
                # take the most represented attribute value among neighbors
                a, nb_occurrence = max(cpt.items(), key=lambda t: t[1])
                predicted_values[node].append(a)
        return predicted_values

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
            # Set dictionary of clusters
            self.nodesclusters[node] = parts[i]
            values.append(parts[i])
            i += 1

        print("Dictionary:", self.nodesclusters)

        nx.draw_networkx_nodes(g, pos, cmap=plt.get_cmap('jet'), node_color=values)
        #### lo comento para que corra mas rapido

        #nx.draw_networkx_edges(g, pos)
        #nx.draw_networkx_labels(g, pos)

        #plt.show()
        #plt.savefig('foo.png')
        #pylab.close()
        #del fig

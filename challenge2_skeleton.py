# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:09:11 2017

@author: cbothore
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np
import pickle
from collections import Counter
from generateGraphPartition import GenerateGraphPartition


def naive_method(graph, empty, attr):
    """   Predict the missing attribute with a simple but effective
    relational classifier. 
    
    The assumption is that two connected nodes are 
    likely to share the same attribute value. Here we chose the most frequently
    used attribute by the neighbors
    
    Parameters
    ----------
    graph : graph
       A networkx graph
    empty : list
       The nodes with empty attributes 
    attr : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values.

    Returns
    -------
    predicted_values : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node (from empty), value is a list of attribute values. Here 
       only 1 value in the list.
     """
    predicted_values = {}
    for n in empty:
        nbrs_attr_values = []
        for nbr in graph.neighbors(n):
            if nbr in attr:
                for val in attr[nbr]:
                    nbrs_attr_values.append(val)
        predicted_values[n] = []
        if nbrs_attr_values:  # non empty list
            # count the number of occurrence each value and returns a dict
            cpt = Counter(nbrs_attr_values)
            # take the most represented attribute value among neighbors
            a, nb_occurrence = max(cpt.items(), key=lambda t: t[1])
            predicted_values[n].append(a)
    return predicted_values


def weighted_naive_method(graph, empty, attr):
    """   Predict the missing attribute with a simple but effective
    relational classifier. 
    
    The assumption is that two connected nodes are 
    likely to share the same attribute value. Here we chose the most frequently
    used attribute by the neighbors
    
    Parameters
    ----------
    graph : graph
       A networkx graph
    empty : list
       The nodes with empty attributes 
    attr : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values.

    Returns
    -------
    predicted_values : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node (from empty), value is a list of attribute values. Here 
       only 1 value in the list.
     """
    predicted_values = {}
    for n in empty:
        nbrs_attr_values = []
        for nbr in graph.neighbors(n):
            if nbr in attr:
                for val in attr[nbr]:
                    nbrs_attr_values.append(val)
        predicted_values[n] = []
        if nbrs_attr_values:  # non empty list
            # count the number of occurrence each value and returns a dict
            cpt = Counter(nbrs_attr_values)
            # take the most represented attribute value among neighbors
            a, nb_occurrence = max(cpt.items(), key=lambda t: t[1])
            predicted_values[n].append(a)
    return predicted_values


def evaluation_accuracy(groundtruth, pred):
    """    Compute the accuracy of your model.

     The accuracy is the proportion of true results.

    Parameters
    ----------
    groundtruth :  : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values.
    pred : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values. 

    Returns
    -------
    out : float
       Accuracy.
    """
    true_positive_prediction = 0
    for p_key, p_value in pred.items():
        if p_key in groundtruth:
            # if prediction is no attribute values, e.g. [] and so is the groundtruth
            # May happen
            if not p_value and not groundtruth[p_key]:
                true_positive_prediction += 1
            # counts the number of good prediction for node p_key
            # here len(p_value)=1 but we could have tried to predict more values
            true_positive_prediction += len([c for c in p_value if c in groundtruth[p_key]])
            # no else, should not happen: train and test datasets are consistent
    return true_positive_prediction * 100 / sum(len(v) for v in pred.values())


def kClicMethod(graph, empty_nodes, attr, location, college, employer, location_predictions=None, college_predictions=None):
    nb_empty_neigh = {}
    for node in empty_nodes:
        nb = 0
        for nbr in graph.neighbors(node):
            if nbr in empty_nodes:
                nb += 1
                nb_empty_neigh[node] = nb
    newAttr = attr


    # Dictionnary { node : number of empty neighbors }
    predicted_values = {}
    print("a ver:", nb_empty_neigh)
    #for n in empty_nodes:
    for n in sorted(nb_empty_neigh, key=lambda t: t[1]):
        nbrs_attr_values = []
        for nbr in graph.neighbors(n):
            if nbr in attr:  # si vecino tiene attr cargado
                for val in attr[nbr]:
                    nbrs_attr_values.append(val)
                    if location_predictions is not None:
                        if location_predictions[n] == location[nbr]:
                            #print("entre")
                            nbrs_attr_values.append(val)

                    if college_predictions is not None:
                        if college_predictions[n] == college[nbr]:
                            #print("coincidio algo")
                            nbrs_attr_values.append(val)
                        """
                        if location_predictions[n] == location[nbr] and college_predictions[n] == college[nbr]:
                            print("coincidio todo")
                            nbrs_attr_values.append(val)
                            nbrs_attr_values.append(val)
                            nbrs_attr_values.append(val)
                        #else:
                            #print("a ver predicion: ", location_predictions[n])
                            #print("a ver el vecino: ", location[nbr])
                            #nbrs_attr_values.append(val)
                    """
        #print("a ver aue tiene sesto: ", nbrs_attr_values)
        predicted_values[n] = []

        if nbrs_attr_values:  # non empty list
            # count the number of occurrence each value and returns a dict
            cpt = Counter(nbrs_attr_values)
            # take the most represented attribute value among neighbors
            a, nb_occurrence = max(cpt.items(), key=lambda t: t[1])
            #print("a ver que tiene: ", a)
            predicted_values[n].append(a)
            #print("a ver que tiene esto: ", predicted_values[n])

    return predicted_values

# load the graph
G = nx.read_gexf("mediumLinkedin.gexf")
print("Nb of users in our graph: %d" % len(G))

# load the profiles. 3 files for each type of attribute
# Some nodes in G have no attributes
# Some nodes may have 1 attribute 'location'
# Some nodes may have 1 or more 'colleges' or 'employers', so we
# use dictionaries to store the attributes
college = {}
location = {}
employer = {}
# The dictionaries are loaded as dictionaries from the disk (see pickle in Python doc)
with open('mediumCollege_60percent_of_empty_profile.pickle', 'rb') as handle:
    college = pickle.load(handle)
with open('mediumLocation_60percent_of_empty_profile.pickle', 'rb') as handle:
    location = pickle.load(handle)
with open('mediumEmployer_60percent_of_empty_profile.pickle', 'rb') as handle:
    employer = pickle.load(handle)

print("Nb of users with one or more attribute college: %d" % len(college))
print("Nb of users with one or more attribute location: %d" % len(location))
print("Nb of users with one or more attribute employer: %d" % len(employer))

# here are the empty nodes for whom your challenge is to find the profiles
empty_nodes = []
with open('mediumRemovedNodes_60percent_of_empty_profile.pickle', 'rb') as handle:
    empty_nodes = pickle.load(handle)
print("Your mission, find attributes to %d users with empty profile" % len(empty_nodes))

# --------------------- Baseline method -------------------------------------#
# Try a naive method to predict attribute
# This will be a baseline method for you, i.e. you will compare your performance
# with this method
# Let's try with the attribute 'employer'

print()
print("### ### ### ### ### ### ### ### ### ### ### ###")
print()

employer_predictions = naive_method(G, empty_nodes, employer)
groundtruth_employer = {}
with open('mediumEmployer.pickle', 'rb') as handle:
    groundtruth_employer = pickle.load(handle)
result = evaluation_accuracy(groundtruth_employer, employer_predictions)
print("Employer: %f%% of the predictions are true" % result)

college_predictions = naive_method(G, empty_nodes, college)
groundtruth_college = {}
with open('mediumCollege.pickle', 'rb') as handle:
    groundtruth_college = pickle.load(handle)
result = evaluation_accuracy(groundtruth_college, college_predictions)
print("College: %f%% of the predictions are true" % result)

location_predictions = naive_method(G, empty_nodes, location)
groundtruth_location = {}
with open('mediumLocation.pickle', 'rb') as handle:
    groundtruth_location = pickle.load(handle)
result = evaluation_accuracy(groundtruth_location, location_predictions)
print("Location: %f%% of the predictions are true" % result)

print()
print("### ### ### ### ### ### ### ### ### ### ### ###")
print()

# --------------------- Now your turn -------------------------------------#
# Explore, implement your strategy to fill empty profiles of empty_nodes


# and compare with the ground truth (what you should have predicted)
# user precision and recall measures

ggp = GenerateGraphPartition(empty_nodes)
employer_predictions = ggp.predictAttributesByCluster(G, empty_nodes, [location, college], employer)
with open('mediumEmployer.pickle', 'rb') as handle:
    groundtruth_employer = pickle.load(handle)
result = evaluation_accuracy(groundtruth_employer, employer_predictions)
print("Employer nuestro: %f%% of the predictions are true" % result)

college_predictions = ggp.predictAttributesByCluster(G, empty_nodes, [employer, location], college)
with open('mediumCollege.pickle', 'rb') as handle:
    groundtruth_college = pickle.load(handle)
result = evaluation_accuracy(groundtruth_college, college_predictions)
print("College nuestro: %f%% of the predictions are true" % result)

location_predictions = ggp.predictAttributesByCluster(G, empty_nodes, [employer, college], location)
with open('mediumLocation.pickle', 'rb') as handle:
    groundtruth_location = pickle.load(handle)
result = evaluation_accuracy(groundtruth_location, location_predictions)
print("Location nuestro: %f%% of the predictions are true" % result)

print()
print("### ### ### ### ### ### ### ### ### ### ### ###")
print()

location_predictions = kClicMethod(G, empty_nodes, location, location, college, employer)
with open('mediumLocation.pickle', 'rb') as handle:
    groundtruth_location = pickle.load(handle)
result = evaluation_accuracy(groundtruth_location, location_predictions)
print("Location kClick: %f%% of the predictions are true" % result)

college_predictions = kClicMethod(G, empty_nodes, college, location, college, employer, location_predictions)
with open('mediumCollege.pickle', 'rb') as handle:
    groundtruth_college = pickle.load(handle)
result = evaluation_accuracy(groundtruth_college, college_predictions)
print("College kClick: %f%% of the predictions are true" % result)

employer_predictions = kClicMethod(G, empty_nodes, college, location, college, employer, location_predictions, college_predictions)
with open('mediumEmployer.pickle', 'rb') as handle:
    groundtruth_employer = pickle.load(handle)
result = evaluation_accuracy(groundtruth_employer, employer_predictions)
print("Employer kClick: %f%% of the predictions are true" % result)


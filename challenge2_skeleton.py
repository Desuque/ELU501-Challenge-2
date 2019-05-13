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
    for n in sorted(nb_empty_neigh, key=lambda t: t[1]):
        nbrs_attr_values = []
        for nbr in graph.neighbors(n):
            if nbr in attr:  # si vecino tiene attr cargado
                for val in attr[nbr]:
                    nbrs_attr_values.append(val)
                    if location_predictions is not None:
                        if location_predictions[n] == location[nbr]:
                            nbrs_attr_values.append(val)

                    if college_predictions is not None:
                        if college_predictions[n] == college[nbr]:
                            nbrs_attr_values.append(val)

        predicted_values[n] = []

        if nbrs_attr_values:  # non empty list
            # count the number of occurrence each value and returns a dict
            cpt = Counter(nbrs_attr_values)
            # take the most represented attribute value among neighbors
            a, nb_occurrence = max(cpt.items(), key=lambda t: t[1])
            predicted_values[n].append(a)

    return predicted_values


def getInfluencers(graph, location, location_predictions=None):
    influencers = []
    # Hardcoded area
    bay_area = ['san francisco bay area']

    for n in graph.nodes():
        node_weight = 0
        for ngb in graph.neighbors(n):
            if ngb in location.keys():
                if location[ngb] == bay_area:
                    node_weight += 1  # Location weight
            if location_predictions is not None:
                if ngb in location_predictions.keys():
                    if location_predictions[ngb] == bay_area:
                        node_weight += 0.5  # Prediction location weight

        if node_weight is not 0:
            influencers.append((n, node_weight))

    influencers = sorted(influencers, key=lambda t: t[1], reverse=True)
    return influencers[:5]

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
print("NQIVE INFLUENCERS METOD:", getInfluencers(G, location, location_predictions))
print()

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
print("METIS INFLUENCERS METOD:", getInfluencers(G, location, location_predictions))
print()

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

print()
print("KCLICK INFLUENCERS METOD:", getInfluencers(G, location, location_predictions))
print()

print()
print("GROUND REAL INFLUENCERS METOD:", getInfluencers(G, groundtruth_location))
print()

"""
CODIGO A TESTEAR
def x_init(G):
    from random import randint
    xValues={}
    for ​ node​ ​ in ​ G.nodes :
        if ​ node​ ​ not in ​ xValues :
            xValues[node]={}
        for ​ n ​ ​ in ​ G.neighbors(node):
            x = randint(​ 1 ​ , ​ 2 ​ )
            if ​ n ​ ​ not in ​ xValues :
                xValues[n]={}
                xValues[node].update({n:x})
                xValues[n].update({node:x})
            else ​ :
                if ​ node​ ​ not in ​ xValues[n]:
                    xValues[n].update({node:x})
                    xValues[node].update({n:x})
    return ​ xValues

def f_init​(G, empty_nodes, location, employer, college):
        fVectors={}
        locations=getallpossible(location)
        colleges=getallpossible(college)
        employers=getallpossible(employer)
        totLen=​ len​ (locations)+​ len​ (colleges)+​ len​ (employers)
        indC=dict(​ zip​ (colleges,[i​ ​ for ​ i ​ ​ in ​ range​ ( ​ 0 ​ , ​ len​ (colleges))]))
        invC=dict(​ zip​ ([i​ ​ for ​ i ​ ​ in ​ range​ ( ​ 0 ​ , ​ len​ (colleges))],colleges))
        indE=dict(​ zip​ (employers,[i​ ​ for ​ i ​ ​ in
    range​ ( ​ len​ (colleges)​ , ​ len​ (colleges)+​ len​ (employers))]))
        invE=dict(​ zip​ ([i​ ​ for ​ i ​ ​ in
    range​ ( ​ len​ (colleges)​ , ​ len​ (colleges)+​ len​ (employers))]​ , ​ employers))
        indL=dict(​ zip​ (locations,[i​ ​ for ​ i ​ ​ in
    range​ ( ​ len​ (colleges)+​ len​ (employers),totLen)]))
        invL=dict(​ zip​ ([i​ ​ for ​ i ​ ​ in
    range​ ( ​ len​ (colleges)+​ len​ (employers),totLen)],locations))
        ind={**indC, **indE, **indL}
        inv={**invC, **invE, **invL}
        omega1=[​ 0 ​ for ​ i ​ ​ in ​ range​ ( ​ len​ (colleges))]
        11omega1.extend([​ 1 ​ for ​ i ​ ​ in ​ range​ ( ​ len​ (employers))])
        omega1.extend([​ 1 ​ for ​ i ​ ​ in ​ range​ ( ​ len​ (locations))])
        omega2=[​ 1 ​ for ​ i ​ in ​ range​ ( ​ len​ (colleges))]
        omega2.extend([​ 0 ​ for ​ i ​ ​ in ​ range​ ( ​ len​ (employers))])
        omega2.extend([​ 1 ​ for ​ i ​ ​ in ​ range​ ( ​ len​ (locations))])
        for ​ node​ ​ in ​ G.nodes :
            if ​ node​ ​ not in ​ empty_nodes :
                fVectors[node]=np.zeros((​ 1 ​ , ​ totLen)).tolist()[0]
                if ​ node​ ​ in ​ location :
                    for ​ loc​ ​ in ​ location[node]:
                        fVectors[node][ind[loc]]=​ 1
                if ​ node​ ​ in ​ college :
                    for ​ col​ ​ in ​ college[node]:
                        fVectors[node][ind[col]]=​ 1
                if ​ node​ ​ in ​ employer :
                    for ​ emp​ ​ in ​ employer[node]:
                        fVectors[node][ind[emp]]=​ 1
            else ​ :
                fVectors[node]=[​ 0.5 ​ for ​ i ​ ​ in ​ range​ (totLen)]
    return ​ fVectors, omega1, omega2, ind, inv, locations, colleges,employers

def predict​(empty_nodes, location, employer, college, fVectors,colleges, employers, inv):
    pred_l, pred_c, pred_e=[], [], []
    for ​ node​ ​ in ​ empty_nodes :
        fVec=fVectors[node]
        colVect=fVec[:​ len​ (colleges)]
        empVect=fVec[​ len​ (colleges):​ len​ (colleges)+​ len​ (employers)]
        locVect=fVec[​ len​ (colleges)+​ len​ (employers):]
        bestColleges=np.argwhere(colVect ==
        np.amax(colVect)).flatten().tolist()
        bestEmployers = np.argwhere(empVect ==
        np.amax(empVect)).flatten().tolist()
        bestLocations= np.argwhere(locVect ==
        np.amax(locVect)).flatten().tolist()
        if ​ len​ (bestColleges)<​ 2 ​ :
            pred_c[node] = []
            for ​ col​ ​ in ​ bestColleges :
                pred_c[node].append(inv[col])
        if ​ len​ (bestEmployers)<​ 5 ​ :
            pred_e[node] = []
            for ​ emp​ ​ in ​ bestEmployers :
                pred_e[node].append(inv[emp+len(colleges)])
        if ​ len​ (bestLocations)<​ 2 ​ :
            pred_l[node] = []
            for ​ loc​ ​ in ​ bestLocations:
                pred_l[node].append(inv[loc+len(colleges)+len(employers)])
   return ​ pred_l, pred_e, pred_c

def coprofiling​(G, empty_nodes, location, employer, college):
    location2, employer2, college2, newempty= location, employer, college,empty_nodes
    fVectors, omega1, omega2, ind, inv, locations, colleges, employers = f_init(G,newempty, location2, employer2,college2)
    xValues = x_init(G)
    nbnode =​ ​ 0
    for ​ xm​ ​ in ​ range ​ ( ​ 3 ​ ) :
        for ​ empty_node​ ​ in ​ newempty:
            print​ ( ​ "Ongoing : %d / %d" ​ % (nbnode, len(newempty)))
            fVectorsnext = fVectors
            circle1 = []
            circle2 = []
            for ​ num​ ​ in ​ range(​ 30​ ):
                # print("Iteration %d sur 10"%num)
                # UPDATE f
                for ​ neigh​ ​ in ​ G.neighbors(empty_node):
                    if ​ xValues[neigh][empty_node] == ​ 1 ​ :
                        circle1.append(neigh)
                    else​ :
                        circle2.append(neigh)
                for ​ neigh​ ​ in ​ circle1:
                    if ​ neigh​ ​ in ​ newempty:
                    L =​ ​ len​ (omega1)
                    for ​ i ​ ​ in​ range(L):
                        if ​ omega1[i] ==​ ​ 1 ​ :
                            fVectorsnext[neigh][i] = fVectors[empty_node][i]
                            for ​ n2​ ​ in ​ circle1:
                                if ​ n2 != neigh:
                                    fVectorsnext[neigh][i] += fVectors[n2][i]
                            fVectorsnext[neigh][i] /= ​ 1 ​ + ​ ​ len​ (circle1)
                for ​ neigh​ ​ in ​ circle2:
                    if ​ neigh​ ​ in ​ newempty:
                    L =​ ​ len​ (omega2)
                    for ​ i ​ ​ in ​ range​ (L):
                        if ​ omega2[i] ==​ ​ 1 ​ :
                            fVectorsnext[neigh][i] = fVectors[empty_node][i]
                            for ​ n2​ ​ in ​ circle2:
                                if ​ n2 != neigh:
                                    fVectorsnext[neigh][i] += fVectors[n2][i]
                            fVectorsnext[neigh][i] /=​ ​ 1 ​ + ​ ​ len​ (circle2)
                for ​ i ​ ​ in ​ range​ (L):
                    den =​ ​ 1
                    if ​ omega1[i] ==​ ​ 1 ​ :
                        den +=​ ​ len​ (omega1)
                        for ​ neigh​ ​ in ​ circle1:
                            fVectorsnext[empty_node][i] += fVectorsnext[neigh][i]
                    if ​ omega2[i] ==​ ​ 1 ​ :
                        den +=​ ​ len​ (omega2)
                        for ​ neigh​ ​ in ​ circle2:
                            fVectorsnext[empty_node][i] += fVectorsnext[neigh][i]
                    fVectorsnext[empty_node][i] /= den
                fVectors = fVectorsnext
                # UPDATE X
                xValuesnext = xValues
                for ​ neigh​ ​ in ​ G.neighbors(empty_node):
                    soust = [fi - f0​ ​ for ​ (fi, f0)​ ​ in ​ zip​ (fVectors[neigh],fVectors[empty_node])]
                    a = -​ 0.7 ​ * ((np.dot(omega1, soust)) **​ ​ 2 ​ )
                    b =​ ​ - ​ 0.7 ​ * ((np.dot(omega2, soust)) **​ ​ 2 ​ )
                    for ​ nj​ ​ in ​ circle1:
                        soust = [fi - fj​ ​ for ​ (fi, fj)​ ​ in ​ zip​ (fVectors[neigh],fVectors[nj])]
                        a +=​ ​ 1 ​ - ​ ​ 0.7 ​ * ((np.dot(omega1, soust)) **​ ​ 2 ​ )
                    for ​ nj​ ​ in ​ circle2:
                        soust = [fi - fj​ ​ for ​ (fi, fj)​ ​ in ​ zip​ (fVectors[neigh],fVectors[nj])]
                        b +=​ ​ 1 ​ -​ ​ 0.7 ​ * ((np.dot(omega2, soust)) ** ​ 2 ​ )
                    if ​ neigh​ ​ not in ​ empty_nodes:
                        a += -​ 7 ​ * ((np.dot(omega1, fVectors[neigh]) -​ ​ 1 ​ ) ** ​ 2 ​ )
                        b += -​ 7 ​ * ((np.dot(omega2, fVectors[neigh]) -​ ​ 1 ​ ) **​ ​ 2 ​ )
                    if ​ a > b:
                        xValuesnext[neigh][empty_node] =​ ​ 1
                        xValuesnext[empty_node][neigh] =​ ​ 1
                    else​ :
                        xValuesnext[neigh][empty_node] =​ ​ 2
                        xValuesnext[empty_node][neigh] =​ ​ 2
                xValues = xValuesnext
            nbnode +=​ ​ 1
            predLoc, predEmp, predCol = predict(empty_nodes, location, employer,college, fVectors, colleges, employers, inv)
            if ​ predLoc :
                print​ ( ​ "Location ("​ , ​ len​ (predLoc.items())​ , ​ ") : "​ ,evaluation_accuracy(groundtruth_location, predLoc))
            if ​ predEmp :
                print​ ( ​ "Employers ("​ , ​ len​ (predEmp.items())​ , ​ ") : "​ ,evaluation_accuracy(groundtruth_employer, predEmp))
            if ​ predCol :
                print​ ( ​ "Colleges ("​ , ​ len​ (predCol.items())​ , ​ ") : "​ ,evaluation_accuracy(groundtruth_college, predCol))
   return ​ xValues, fVectors, predLoc, predEmp, predCol
"""
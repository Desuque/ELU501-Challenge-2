#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:08:56 2019

@author: g18quint
"""
from generateGraphPartition import GenerateGraphPartition
import pickle


def mainTest():
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


    print("hola")
    ggp = GenerateGraphPartition(empty_nodes)
    ggp.createEdgesGraph()
    print("hola")


if __name__ == '__main__':
    mainTest()

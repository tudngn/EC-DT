# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:52:12 2019

@author: z5095790
"""
import numpy as np
import copy
import pickle
import os
from keras.models import load_model


class Node:
    """Binary tree with Ture and False Branches"""
    def __init__(self, col=-1, value = None, parentID = None, ID = None, branch=None, results=None, numSamples=0, memory=None, leaf =0):
        self.col = col
        self.value = value
        self.parentID = parentID
        self.ID = ID
        self.branch = branch
        self.results = results #
        self.numSamples = numSamples # stat of the number of training samples flow through that leaf
        self.memory = memory # store samples fall into that leaf
        self.leaf = leaf # FALSE for nodes, TRUE for leaves


def load_weight_bias_path(path):

    DNNModel = load_model(path)
    
    weight1 = DNNModel.layers[0].get_weights()[0]	
    biase1 = DNNModel.layers[0].get_weights()[1] #2,4
    
    weight2 = DNNModel.layers[2].get_weights()[0]
    biase2 = DNNModel.layers[2].get_weights()[1] #2,4
    
    weight3 = DNNModel.layers[4].get_weights()[0]
    biase3 = DNNModel.layers[4].get_weights()[1] #2,4
    
    weight = [weight1, weight2, weight3]
    bias = [biase1, biase2, biase3]
    
    return(weight, bias)
    
    
def countLeaves(tree,showID = False):
    count = 0
    leafID_list = []
    for i in range(0,len(tree)):
        if tree[i].leaf == 1:
            count += 1
            leafID_list.append(tree[i].ID)
    if showID:
        return (count,leafID_list)
    return count


def activationByLayer(hidden_nodes,activations):
    activationByLayer = []
    startNode = 0
    for i in range(0, len(hidden_nodes)):
        layer_activation = []
        num_node_layer = hidden_nodes[i]
        for j in range(0,num_node_layer):
            layer_activation.append(activations[startNode+j])
        activationByLayer.append(layer_activation)
        startNode = startNode + num_node_layer
        
    return activationByLayer


def transformWeight(hidden_nodes,activations,weight,bias):
    
    weight_input_to_layers = copy.deepcopy(weight)
    bias_input_to_layers = copy.deepcopy(bias)
    weight_layer_activated = copy.deepcopy(weight)
    for i in range(0,len(activations)):            
        for j in range(0,hidden_nodes[i]):
            if activations[i][j] == 0:
                weight_layer_activated[i+1][j,:] = 0
        weight_input_to_layers[i+1] = np.matmul(weight_input_to_layers[i],weight_layer_activated[i+1])
        bias_input_to_layers[i+1] = np.matmul(bias_input_to_layers[i],weight_layer_activated[i+1]) + bias_input_to_layers[i+1]    
    
    return (weight_input_to_layers,bias_input_to_layers)


def extractRules(tree,hidden_nodes,num_input,num_output,weight,bias):
       
    leaves_list = []
    rule_list = []
    rule_list_txt = []
    
    num_leaves, leaves_list = countLeaves(tree, showID = True)
    num_hidden_layers = len(hidden_nodes)       

    # create a list of names for input and output vectors
    input_name_array = []
    for i in range(0,num_input):
        input_name_array.append('X_'+str(i))
        
    output_name_array = []
    for i in range(0,num_output):
        output_name_array.append('Y_'+str(i))
          
    # generate rules for each leaf  
    num_constraints = np.zeros([num_leaves,1])
    for i in range(0,num_leaves):
        
        leafResult = tree[leaves_list[i]].results
        leafResultByLayer = activationByLayer(hidden_nodes,leafResult)
        weight_input_to_layers, bias_input_to_layers = transformWeight(hidden_nodes,leafResultByLayer,weight,bias)
        
        # rules for activating hidden layers
        rule_txt = 'IF:\n\n'
        rule = np.zeros([tree[leaves_list[i]].col+1,num_input+2])
        startCol = 0
        
        for j in range(0,num_hidden_layers):
            
            for m in range(0,hidden_nodes[j]):
                
                if startCol == tree[leaves_list[i]].col:
                    break
                else:
                
                    for k in range(0,num_input):
                        if k == 0:
                            rule_txt = rule_txt + '(' + str(weight_input_to_layers[j][k,m]) + input_name_array[k] + ')'
                        else:
                            rule_txt = rule_txt + ' + (' + str(weight_input_to_layers[j][k,m]) + input_name_array[k] + ')'
                        rule[startCol,k] = weight_input_to_layers[j][k,m]
                    if leafResultByLayer[j][m] == 1:
                        rule_txt = rule_txt + ' > ' + str(-bias_input_to_layers[j][m]) + "\n"
                        rule[startCol,-1] = 1

                    else:
                        rule_txt = rule_txt + ' <= ' + str(-bias_input_to_layers[j][m]) + "\n"
                        rule[startCol,-1] = -1
                        
                    rule[startCol,num_input] = bias_input_to_layers[j][m]
                    
                    startCol += 1
            
            rule_txt = rule_txt + "THEN hidden layer " + str(j) + " activation is: " + str(leafResultByLayer[j]) + "\n\n"
     
        # rules for decision at output                
        for m in range(0,num_output):
            if num_output == 1:
                rule_txt = rule_txt + 'IF:\n' 
            else:
                result = '\t\t' + output_name_array[j] + ' = softmax('
                for k in range(0,num_input):
                    if k == 0:
                        rule_txt = rule_txt + '(' + str(weight_input_to_layers[-1][k,m]) + input_name_array[k] + ')'                        
                    else:
                        rule_txt = rule_txt + ' + (' + str(weight_input_to_layers[-1][k,m]) + input_name_array[k] + ')'
                    rule[-1,k] = weight_input_to_layers[-1][k,m]
                rule_txt = rule_txt + ' + (' + str(bias_input_to_layers[-1][m]) + ') > ' + str(0) + "\n"
                rule_txt = rule_txt + 'THEN: class = 1, OTHERWISE class = 0.'    
                rule[-1,num_input] = bias_input_to_layers[-1][m]
        rule_list_txt.append(rule_txt)
        rule_list.append(rule)
        num_constraints[i] = len(rule)-1
        
    return (rule_list_txt, rule_list, num_constraints) 


if __name__ == '__main__':

    hidden_nodes = [5,5]
    num_input = 2
    num_output = 4

    Tree_Directory = "./NumericalData/wall-following-2/Saved_Trees/"
    Model_Directory = "./NumericalData/wall-following-2/Model/"
    listdir_PrunedTrees = os.listdir(Tree_Directory)
    listdir_PrunedTrees = listdir_PrunedTrees[0:100]
    listdir_DNNmodel = os.listdir(Model_Directory)
    #load tree, weight, bias
    total_num_constraints = None

    for i in range(0, 10):
        weight, bias = load_weight_bias_path(Model_Directory + listdir_DNNmodel[i])
        with open(Tree_Directory + listdir_PrunedTrees[i], 'rb') as f:
            tree = pickle.load(f)    
        rule_list_txt, rule_list, num_constraints = extractRules(tree,hidden_nodes,num_input,num_output,weight,bias)
        
        if total_num_constraints is None:
            total_num_constraints = num_constraints
        else:
            total_num_constraints = np.vstack([total_num_constraints,num_constraints])
        print("Tree %d extracted." %i)
        
        
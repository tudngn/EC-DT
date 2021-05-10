# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:38:28 2019

@author: z5095790
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 12:04:35 2019

@author: z5095790
"""
 
#import collections
import numpy as np
import pickle
import copy
import os
import winsound
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
#    input_name_array = []
#    for i in range(0,num_input):
#        input_name_array.append('X_'+str(i))
        
#    output_name_array = []
#    for i in range(0,num_output):
#        output_name_array.append('Y_'+str(i))
          
    # generate rules for each leaf    
    for i in range(0,num_leaves):
        
        leafResult = tree[leaves_list[i]].results
        leafResultByLayer = activationByLayer(hidden_nodes,leafResult)
        weight_input_to_layers, bias_input_to_layers = transformWeight(hidden_nodes,leafResultByLayer,weight,bias)
        
        # rules for activating hidden layers
#        rule_txt = 'IF:\n\n'
        rule = np.zeros([tree[leaves_list[i]].col+1,num_input+2])
        startCol = 0
        for j in range(0,num_hidden_layers):
            
            for m in range(0,hidden_nodes[j]):
                
                if startCol == tree[leaves_list[i]].col:
                    break
                else:
                
                    for k in range(0,num_input):
#                        if k == 0:
#                            rule_txt = rule_txt + '(' + str(weight_input_to_layers[j][k,m]) + input_name_array[k] + ')'
#                        else:
#                            rule_txt = rule_txt + ' + (' + str(weight_input_to_layers[j][k,m]) + input_name_array[k] + ')'
                        rule[startCol,k] = weight_input_to_layers[j][k,m]
                    if leafResultByLayer[j][m] == 1:
#                        rule_txt = rule_txt + ' > ' + str(-bias_input_to_layers[j][m]) + "\n"
                        rule[startCol,-1] = 1

                    else:
#                        rule_txt = rule_txt + ' <= ' + str(-bias_input_to_layers[j][m]) + "\n"
                        rule[startCol,-1] = -1
                        
                    rule[startCol,num_input] = bias_input_to_layers[j][m]
                    
                    startCol += 1
            
#            rule_txt = rule_txt + "THEN hidden layer " + str(j) + " activation is: " + str(leafResultByLayer[j]) + "\n\n"
     
        # rules for decision at output                
        for m in range(0,num_output):
            if num_output == 1:
#                rule_txt = rule_txt + 'IF:\n' 
                for k in range(0,num_input):
#                    if k == 0:
#                        rule_txt = rule_txt + '(' + str(weight_input_to_layers[-1][k,m]) + input_name_array[k] + ')'                        
#                    else:
#                        rule_txt = rule_txt + ' + (' + str(weight_input_to_layers[-1][k,m]) + input_name_array[k] + ')'
                    rule[-1,k] = weight_input_to_layers[-1][k,m]
#                rule_txt = rule_txt + ' + (' + str(bias_input_to_layers[-1][m]) + ') > ' + str(0) + "\n"
#                rule_txt = rule_txt + 'THEN: class = 1, OTHERWISE class = 0.'    
                rule[-1,num_input] = bias_input_to_layers[-1][m]
#        rule_list_txt.append(rule_txt)
        if num_output == 1:
            rule_list.append(rule)
        else:
            rule_list.append(rule[1:-1,:])
    return (rule_list_txt, rule_list) 


''' Main: extracting rule list'''
# Initialize parameters
hidden_nodes = [5,5]
num_input = 2
num_output = 4

# Import tree and weights, and extracting rules
# Get the list of tree filenames
tree_Directory = "./NumericalData/wall-following-2/Saved_Trees/Pruned/"
listFile_tree = os.listdir(tree_Directory)
model_Directory = "./NumericalData/wall-following-2/Model/"
listFile_model = os.listdir(model_Directory)
num_constraints_per_tree = []
num_constraints_per_leaf = []

for i in range(0,len(listFile_tree)):
    file_in = tree_Directory + listFile_tree[i]
    with open(file_in, 'rb') as f:
        tree = pickle.load(f)
    weight, bias = load_weight_bias_path(model_Directory + listFile_model[i])
    rule_list_txt, rule_list = extractRules(tree,hidden_nodes,num_input,num_output,weight,bias)

    num_rules = len(rule_list)
    num_constraints = 0
    for j in range(0,len(rule_list)):
        num_constraints = num_constraints + len(rule_list[j])
        
    num_constraints_per_tree.append(num_constraints)
    num_constraints_per_leaf.append(num_constraints/num_rules)

# Compute mean and std    
mean_tree_constraints = np.mean(num_constraints_per_tree)
std_tree_constraints = np.std(num_constraints_per_tree)

mean_leaf_constraints = np.mean(num_constraints_per_leaf)
std_leaf_constraints = np.std(num_constraints_per_leaf)
    
print("leaf_constraints = %.2f +- %.2f " %(mean_leaf_constraints,std_leaf_constraints))
print("tree_constraints = %.2f +- %.2f " %(mean_tree_constraints,std_tree_constraints))

winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
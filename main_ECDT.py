import csv 
#import collections
import numpy as np
import pickle
import datetime
import pandas as pd
import copy
from scipy.special import expit
from keras.models import load_model
from sklearn.utils.extmath import softmax
from sklearn.metrics import mean_squared_error
from secondOrder import secondOrder
from DNN_train import DNN_train
from DNN_train_binary import DNN_train_binary
from read_data_csv import read_data_csv
from read_data_txt import read_data_txt



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


def createNode(DNN_node_ID, parentID=None, parentResult=None, newID=0, isTrue=1, isleaf=0):
    
    if newID == 0: # is root
        return Node(col=DNN_node_ID, value = 0, ID=0)        
    
    if parentID == 0: # child of first node
        return Node(col=DNN_node_ID, value = 0, parentID = parentID, ID=newID, branch = isTrue, results=isTrue, leaf = isleaf)        
    else:
        return Node(col=DNN_node_ID, value = 0, parentID = parentID, ID=newID, branch = isTrue, results=np.hstack([parentResult,isTrue]), leaf = isleaf)        
        
        
def growDecisionTree(hidden_nodes):
        
    num_NN_HDnodes = np.sum(hidden_nodes) 

    tree = []
    currentParents = []
    currentChildren = []
    newID = 0
    
    if num_NN_HDnodes == 0: return tree
        
    DNN_node_ID = 0
    if DNN_node_ID == 0: # fisrt node (root)
        node = createNode(DNN_node_ID)
        tree.append(node)       
        DNN_node_ID = 1
        newID += 1
        currentParents.append(node.ID)
            
    # next layers:
    while (DNN_node_ID <= num_NN_HDnodes):   
        
        for i in range(0,len(currentParents)):
                    
            if DNN_node_ID == num_NN_HDnodes: # to create leaves
                trueLeaf = createNode(DNN_node_ID, parentID = currentParents[i], parentResult = tree[currentParents[i]].results, newID=newID, isTrue=1, isleaf = 1)
                newID += 1
                falseLeaf = createNode(DNN_node_ID, parentID = currentParents[i], parentResult = tree[currentParents[i]].results, newID=newID, isTrue=0, isleaf = 1)
                newID += 1
                tree.append(trueLeaf)
                tree.append(falseLeaf)
            else: 
                trueNode = createNode(DNN_node_ID, parentID = currentParents[i], parentResult = tree[currentParents[i]].results, newID=newID, isTrue=1, isleaf = 0)
                newID += 1
                falseNode = createNode(DNN_node_ID, parentID = currentParents[i], parentResult = tree[currentParents[i]].results, newID=newID, isTrue=0, isleaf = 0)
                newID += 1
                currentChildren.append(trueNode.ID)
                currentChildren.append(falseNode.ID)
                tree.append(trueNode)
                tree.append(falseNode)
        
        currentParents = currentChildren
        currentChildren = []
        DNN_node_ID += 1
    
    return tree


def findTreeRoute(observations, tree, startNodeID, startLayer, stopLayer, alternateLeafActivation = False):
    """Classifies the observations according to the tree."""
    col = 0
    treeRoute = []
    for i in range(startLayer,stopLayer+2):
        if i == startLayer:
            currentNode = tree[startNodeID]
            
            if currentNode.leaf == 1:
                treeRoute.append(currentNode.ID)
                break
            else:                
                v = observations[col]
                if v > currentNode.value: branch = 1
                else: branch = 0
                treeRoute.append(startNodeID)
                col += 1
        
        elif i == stopLayer+1:
            for j in range(currentNode.ID+1,len(tree)): # find linked node for the case
                if tree[j].parentID == currentNode.ID and tree[j].branch == branch:
                    currentNode = tree[j]
                    break
            if currentNode.leaf == 1:
                
                # Choose the counter leaf node
                if alternateLeafActivation:
                    for k in range(0,len(tree)):
                        if tree[k].parentID == currentNode.parentID and tree[k].ID != currentNode.ID:
                            currentNode = tree[k]
                treeRoute.append(currentNode.ID)
                
        else:        
            for j in range(currentNode.ID+1,len(tree)): # find linked node for the case
                if tree[j].parentID == currentNode.ID and tree[j].branch == branch:
                    currentNode = tree[j]
                    break
            if currentNode.leaf == 1:
                treeRoute.append(currentNode.ID)
                break
            else:
                v = observations[col]
                if v > currentNode.value: branch = 1
                else: branch = 0
                treeRoute.append(currentNode.ID)
                col += 1
            
    return (treeRoute, currentNode.ID, currentNode.results[startLayer:stopLayer+1])

def loadCSV(file):
	"""Loads a CSV file and converts all floats and ints into basic datatypes."""
    
	def convertTypes(s):
		s = s.strip()
		try:
			return float(s) if '.' in s else int(s)
		except ValueError:
			return s	

	reader = csv.reader(open(file, 'rt'))
	return [[convertTypes(item) for item in row] for row in reader]


def load_weight_bias(Model):

    DNNModel = Model
    
    weight1 = DNNModel.layers[0].get_weights()[0]	
    biase1 = DNNModel.layers[0].get_weights()[1] #2,4
    
    weight2 = DNNModel.layers[2].get_weights()[0]
    biase2 = DNNModel.layers[2].get_weights()[1] #2,4
    
    weight3 = DNNModel.layers[4].get_weights()[0]
    biase3 = DNNModel.layers[4].get_weights()[1] #2,4
    
    weight = [weight1, weight2, weight3]
    bias = [biase1, biase2, biase3]
    
    return(weight, bias)
    
    
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
    

def tree_prediction(new_data,
                    hidden_nodes,
                    num_output,
                    tree,
                    weight,
                    bias,
                    alternateLeafActivation = False,
                    showTreeRoute = False):
    
    
    # data input to HD1
    num_HD_layers = len(hidden_nodes)
    observations = new_data
    startNodeID = 0
    startLayer = 0
    treeRoute = []

    for i in range(0,num_HD_layers):
        stopLayer = startLayer + hidden_nodes[i] - 1
        layer_input = np.matmul(observations,weight[i]) + bias[i]
        treeRoute_temp, currentNode_ID, layer_activation = findTreeRoute(layer_input, tree, startNodeID, startLayer, stopLayer, alternateLeafActivation)
        observations = np.multiply(layer_input,layer_activation)
        
        # Update variables with new values
        startNodeID = currentNode_ID
        startLayer = stopLayer + 1
        for j in range(0,len(treeRoute_temp)):
            treeRoute.append(treeRoute_temp[j])
            
    # output layer
    output = np.matmul(observations,weight[-1]) + bias[-1]
    
    #classification problem    
    if num_output == 1:
        output = expit(np.resize(output,[1,num_output])) # sigmoid activation
    else:
        output = softmax(np.resize(output,[1,num_output])) # softmax activation  '''     
    
    if showTreeRoute:    
        return (treeRoute, output)
    else:
        return output
     

def treeTrain(data, hidden_nodes,num_output, tree, weight, bias):
    
    num_instances = data.shape[0]
    temp_tree = copy.deepcopy(tree)
    # Pass data through tree
    for i in range(0,num_instances):
        instance = data[i,:]
        treeRoute, output = tree_prediction(instance[0:len(instance)-1], hidden_nodes, num_output, temp_tree, weight, bias, showTreeRoute = True)
        
        if num_output == 1:
            dataToMemory = np.hstack([np.reshape(instance,[1,len(instance)]),np.reshape(np.round(output),[1,1])])
        else:
            dataToMemory = np.hstack([np.reshape(instance,[1,len(instance)]),np.reshape(np.argmax(output),[1,1])])
        for j in range(0,len(treeRoute)):
            temp_tree[treeRoute[j]].numSamples += 1
            if temp_tree[treeRoute[j]].memory is None:
                temp_tree[treeRoute[j]].memory = dataToMemory
            else:
                temp_tree[treeRoute[j]].memory = np.vstack([temp_tree[treeRoute[j]].memory,dataToMemory])
    
    return temp_tree


def treeTest(data, hidden_nodes, num_output, tree, weight, bias): 
    
    num_instances = data.shape[0]
    num_leaves, leaves_list = countLeaves(tree, showID = True)
    memory_list = [None]*num_leaves
    
    # Pass data through tree
    for i in range(0,num_instances):
        instance = data[i,:]
        treeRoute, output = tree_prediction(instance[0:len(instance)-1], hidden_nodes, num_output, tree, weight, bias, showTreeRoute = True)
        if num_output == 1:
            dataToMemory = np.hstack([np.reshape(instance,[1,len(instance)]),np.reshape(np.round(output),[1,1])])
        else:
            dataToMemory = np.hstack([np.reshape(instance,[1,len(instance)]),np.reshape(np.argmax(output),[1,1])])
        for j in range(0,len(treeRoute)):
            if  tree[treeRoute[j]].leaf == 1:
                leafID = tree[treeRoute[j]].ID
                idx = leaves_list.index(leafID)
                           
                if memory_list[idx] is None:
                    memory_list[idx] = dataToMemory
                else:
                    memory_list[idx] = np.vstack([memory_list[idx],dataToMemory])
    
    # Compute the accuracy at each node
    leaves_acc = np.zeros(num_leaves)
    total_misClass_count = 0
    misClass_count = 0
    for i in range(0,num_leaves):
        if num_output == 1:
            misClass_count = np.sum(np.abs(memory_list[i][:,-2] - memory_list[i][:,-1]))
        else:
            for j in range(0, memory_list[i].shape[0]):
                if (memory_list[i][:,-2] != memory_list[i][:,-1]):
                    misClass_count += 1
                
        leaves_acc[i] = 1 - misClass_count/memory_list[i].shape[0]
        total_misClass_count += misClass_count 
        
    # Compute the tree's accuracy
    tree_acc = 1 - total_misClass_count/num_instances
    
    return (leaves_list, leaves_acc, tree_acc)

             
        
def prune(trainedTree,hidden_nodes, num_output, weight, bias, misClass_threshold = 0.1): 
    
    
    ## Prune tree according to the increases of MSE values and classification error compared to a threshold
    # find leaves
    tree = copy.deepcopy(trainedTree)
    for i in range(0,len(tree)):
        if tree[i].leaf == 1:
            startLeafID = i
            break
    
    num_pair_leaves = int((len(tree) - startLeafID)/2)
    
    # consider each pair of leaves
    for i in range(0,num_pair_leaves):
        trueLeaf = tree[startLeafID]
        falseLeaf = tree[startLeafID+1]
        
        if trueLeaf.numSamples == 0 or falseLeaf.numSamples == 0:
            tree[trueLeaf.parentID].leaf = 1
            tree[trueLeaf.parentID].results = tree[trueLeaf.ID].results
            if falseLeaf.numSamples != 0:
                tree[trueLeaf.parentID].results = tree[falseLeaf.ID].results
            tree[trueLeaf.ID].leaf = 0
            tree[falseLeaf.ID].leaf = 0

        else:
            tempMemorywithClasses = tree[trueLeaf.parentID].memory
            if num_output == 1:
                currentMisClass = np.sum(np.abs(tempMemorywithClasses[:,-2] - tempMemorywithClasses[:,-1]))/tempMemorywithClasses.shape[0]
            else:
                MisClassCount = 0
                for j in range(0,tempMemorywithClasses.shape[0]):
                    if (tempMemorywithClasses[j,-2] != tempMemorywithClasses[j,-1]):
                        MisClassCount += 1
                currentMisClass = MisClassCount/tempMemorywithClasses.shape[0]
                        
            
            if trueLeaf.numSamples >= falseLeaf.numSamples:
                for j in range(0,falseLeaf.numSamples):
                    instance = falseLeaf.memory[j,:]
                    output = tree_prediction(instance[0:-2], hidden_nodes, num_output, tree, weight, bias, alternateLeafActivation = True)
                    if num_output == 1:
                        falseLeaf.memory[j,-1] = np.round(output)
                    else:
                        falseLeaf.memory[j,-1] = np.argmax(output)
            else:
                for j in range(0,trueLeaf.numSamples):
                    instance = trueLeaf.memory[j,:]
                    output = tree_prediction(instance[0:-2], hidden_nodes, num_output, tree, weight, bias, alternateLeafActivation = True)
                    if num_output == 1:
                        trueLeaf.memory[j,-1] = np.round(output)
                    else:
                        trueLeaf.memory[j,-1] = np.argmax(output)
                    
            tempMemorywithClasses = np.vstack([trueLeaf.memory,falseLeaf.memory])
            if num_output == 1:
                newMisClass = np.sum(np.abs(tempMemorywithClasses[:,-2] - tempMemorywithClasses[:,-1]))/tempMemorywithClasses.shape[0]
            else:
                MisClassCount = 0
                for j in range(0,tempMemorywithClasses.shape[0]):
                    if (tempMemorywithClasses[j,-2] != tempMemorywithClasses[j,-1]):
                        MisClassCount += 1
                newMisClass = MisClassCount/tempMemorywithClasses.shape[0]
    
            if (newMisClass - currentMisClass) < misClass_threshold:
                tree[trueLeaf.parentID].memory = tempMemorywithClasses                
                tree[trueLeaf.parentID].leaf = 1
                if trueLeaf.numSamples >= falseLeaf.numSamples:
                    tree[trueLeaf.parentID].results = tree[trueLeaf.ID].results
                else:
                    tree[trueLeaf.parentID].results = tree[falseLeaf.ID].results
                tree[trueLeaf.ID].leaf = 0
                tree[falseLeaf.ID].leaf = 0
    
    # delete all nodes that have not been used 
    unused_branch = 1
    while(unused_branch):
        unused_branch = 0           
        for i in range(0,len(tree)):
            for j in range(0,len(tree)):
                if j != i and tree[i].parentID == tree[j].parentID and tree[i].numSamples == 0 and tree[j].numSamples == 0 and tree[i].leaf == 1 and tree[j].leaf == 1:
                    tree[i].leaf = 0
                    tree[j].leaf = 0                    
                    tree[tree[i].parentID].leaf = 1
                    tree[tree[i].parentID].results = tree[min([i,j])].results
                    unused_branch = 1
                elif j != i and tree[i].parentID == tree[j].parentID and tree[j].numSamples == 0 and tree[i].numSamples != 0 and tree[j].leaf == 1:
                    tree[j].leaf = 0
                    tree[tree[i].parentID].results = tree[i].results
                    unused_branch = 1 
                       
    return tree


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
        
#    output_name_array = []
#    for i in range(0,num_output):
#        output_name_array.append('Y_'+str(i))
          
    # generate rules for each leaf    
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
#            else:
#                result = '\t\t' + output_name_array[j] + ' = softmax('
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
    return (rule_list_txt, rule_list) 
    
    
def read_data(numTrain = 10):
    
     # Sampling data 
#    Data = read_data_txt()
    Data = read_data_csv()
#    Data = secondOrder()
    
    num_instances = Data.shape[0]  
    NumTest = round(0.2*num_instances)
    
    # split data to train and test
    np.random.shuffle(Data)
    testingData = Data[0:NumTest,:]
    trainingData_original = Data[NumTest:num_instances,:]
    
    trainingSets = np.zeros([trainingData_original.shape[0],trainingData_original.shape[1],numTrain])
    
    for i in range(0,numTrain):
        trainingData = copy.deepcopy(trainingData_original)
        np.random.shuffle(trainingData)
        trainingSets[:,:,i] = trainingData
    return (trainingSets, testingData)
    

if __name__ == '__main__':

    ''' Load DNN model's params or train new DNN''' 
    
    directory = "NumericalData/" # Directory for all data
    data_name = "wall-following-2" # Directory for specific data
    num_output = 4
    hidden_nodes = [5,5]
    num_folds = 10
       
    for fold in range(0,num_folds):
        now = datetime.datetime.now()
        header_acc = ["DNN_Accuracy","Tree_Accuracy", "Pruned_Tree_Accuracy","Tree_Size","Pruned_tree_size"]
        filename_acc = directory + data_name + "/Accuracy_data_fold" + str(fold) + "_config_" + str(hidden_nodes[0]) + "-" + str(hidden_nodes[1]) + "_" + now.strftime("%Y%m%d-%H%M") + ".csv"
        
        with open(filename_acc, 'w') as f:
            pd.DataFrame(columns = header_acc).to_csv(f,encoding='utf-8', index=False, header = True)
        
        numTrain = 10
        results = np.zeros([numTrain,5])
        # Read new data in
        trainingSets, testingData = read_data(numTrain) 
        num_samples_train = trainingSets.shape[0]
        num_samples = testingData.shape[0]        
        filename_train_data = directory + data_name + "/Data/Training_data_fold" + str(fold) + "_config_" + str(hidden_nodes[0]) + "-" + str(hidden_nodes[1]) + "_" + now.strftime("%Y%m%d-%H%M") + ".npy"
        filename_test_data = directory + data_name + "/Data/Testing_data_fold" + str(fold) + "_config_" + str(hidden_nodes[0]) + "-" + str(hidden_nodes[1]) + "_" + now.strftime("%Y%m%d-%H%M") + ".npy"
        np.save(filename_train_data,trainingSets)
        np.save(filename_test_data,testingData)
        
        
        for k in range(0,numTrain):
            
            tree_filename = directory + data_name + "/Saved_Trees/Tree" + str(fold) + "_trial" + str(k) + "_" + str(hidden_nodes[0]) + "-" + str(hidden_nodes[1]) + "_" + now.strftime("%Y%m%d-%H%M") + ".pkl"
            pruned_tree_filename = directory + data_name + "/Saved_Trees/PruneTree" + str(fold) + "_trial" + str(k) + "_" + str(hidden_nodes[0]) + "-" + str(hidden_nodes[1]) + "_" + now.strftime("%Y%m%d-%H%M") + ".pkl"
            #dnn, dnn_accuracy = DNN_train(num_output,hidden_nodes,trainingSets[:,:,k],testingData)
            dnn, dnn_accuracy = DNN_train_binary(hidden_nodes,trainingSets[:,:,k],testingData)
            dnn.model.save(directory + data_name + "/Model/DNNmodel" + str(fold) + "_trial" + str(k) + "_" + str(hidden_nodes[0]) + "-" + str(hidden_nodes[1]) + "_" + now.strftime("%Y%m%d-%H%M") + ".h5")
            results[k,0] = dnn_accuracy
           
            # print results:
            print("Accuracy of DNN model @ fold %d - trial %d = %2f" %(fold,k,dnn_accuracy))
       
            # Extract params from DNN to build trees
            weight, bias = load_weight_bias(dnn.model) 
            decisionTree = growDecisionTree(hidden_nodes)
            results[k,3] = countLeaves(decisionTree) 
            
            # pass data to store in tree
            trainedTree = treeTrain(trainingSets[:,:,k], hidden_nodes,num_output, decisionTree, weight, bias)                               
                        
            '''For predicting multiple samples - no pruning'''
            prediction = np.zeros([num_samples])            
            for i in range(0,num_samples):
                output_values = tree_prediction(testingData[i,0:-1],
                                                hidden_nodes,
                                                num_output,
                                                trainedTree,
                                                weight,
                                                bias)
                
                if num_output == 1:
                    prediction[i] = np.round(output_values)
                else: 
                    prediction[i] = np.argmax(output_values)
                
            # Accuracy
            count = 0
            for i in range(0,num_samples):
                if prediction[i] == testingData[i,-1]:
                    count += 1
            acc = count/num_samples
            
            print("Accuracy of trees @ fold %d - trial %d = %2f" %(fold,k,acc))
            results[k,1] = acc
            
            # Save trained_trees
            with open(tree_filename, 'wb') as f1:
                pickle.dump(trainedTree, f1)
        
            # Prune trees based on training data
            prunedTree = prune(trainedTree, hidden_nodes, num_output, weight, bias) 
            
        
            '''Test pruned trees'''
            results[k,4] = countLeaves(prunedTree)
            
            '''For predicting multiple samples - pruned tree'''
            prediction = np.zeros([num_samples])
            for i in range(0,num_samples):
                output_values = tree_prediction(testingData[i,0:-1],
                                hidden_nodes,
                                num_output,
                                prunedTree,
                                weight,
                                bias)
                
                if num_output == 1:
                    prediction[i] = np.round(output_values)
                else: 
                    prediction[i] = np.argmax(output_values)
                
            # Accuracy
            count = 0
            for i in range(0,num_samples):
                if prediction[i] == testingData[i,-1]:
                    count += 1
            acc = count/num_samples
            
            print("Accuracy of pruned trees @ fold %d - trial %d = %2f" %(fold,k,acc))
            results[k,2] = acc
            
            #save pruned trees
            with open(pruned_tree_filename, 'wb') as f2:
                pickle.dump(prunedTree, f2)
        
        with open(filename_acc, 'a') as f:
            pd.DataFrame(results).to_csv(f,encoding='utf-8', index=False, header = False)
        
        
        
    
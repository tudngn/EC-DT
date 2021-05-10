# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:23:39 2019

@author: Lab User
"""

from DeepNet_binary import DeepNet_binary
import numpy as np

def DNN_train_binary(Num_Hd_nodes,trainingSets,testingData):

    # Initialize new network
    num_attributes = trainingSets.shape[1]-1
    NumTest = testingData.shape[0]
    
    dnn = DeepNet_binary(num_attributes,Num_Hd_nodes,1)
    # train model
    dnn.model.fit(trainingSets[:,0:num_attributes], trainingSets[:,-1].astype(int), epochs=500, batch_size=128, shuffle=False)
    # evaluate model
#    dnn.model.evaluate(testingData[:,1:Data.shape[1]], testingData[:,0].astype(int), batch_size=128)
    # test model
    model_output = dnn.model.predict(testingData[:,0:num_attributes])
    prediction = np.round(model_output)
    
    # Accuracy
    count = 0
    for i in range(0,NumTest):
        if prediction[i] == testingData[i,-1]:
            count += 1
    acc = count/NumTest

    return(dnn,acc)
    


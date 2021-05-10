# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:23:39 2019

@author: Lab User
"""

import keras
from DeepNet import DeepNet
import numpy as np

def DNN_train(classes,Num_Hd_nodes,trainingData,testingData):
    

# Initialize new network
    num_attributes = trainingData.shape[1]-1
    NumTest = testingData.shape[0]
   
    dnn = DeepNet(num_attributes,Num_Hd_nodes,classes)
    
    # Convert labels to categorical one-hot encoding
    one_hot_labels_training = keras.utils.to_categorical(trainingData[:,-1], num_classes=classes)

    # train model
    dnn.model.fit(trainingData[:,0:num_attributes], one_hot_labels_training, epochs=500, batch_size=128)
    
    # test model
    model_output = dnn.model.predict(testingData[:,0:num_attributes])
    prediction = np.argmax(model_output, axis = 1)

    # Accuracy
    count = 0
    for i in range(0,NumTest):
        if prediction[i] == testingData[i,-1]:
            count += 1
    acc = count/NumTest

    return(dnn,acc)

    
    



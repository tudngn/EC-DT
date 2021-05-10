# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:19:36 2019

@author: Lab User
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np


# Constants
LR = 0.0001

# Deep Network
class DeepNet: 
   
    def __init__(
            self,
            input_dim,
            num_HD_nodes,
            action_space,
            sess=None
            
    ):
      self.input_dim = input_dim
      self.num_HD_nodes = num_HD_nodes
      self.action_space = action_space
      self.learning_rate = LR
            
      #Creating networks
      self.model = self.create_model()

      
    def create_model(self):
      model = Sequential()
      model.add(Dense(self.num_HD_nodes[0], input_dim=self.input_dim))
      model.add(Activation('relu'))
      model.add(Dense(self.num_HD_nodes[1]))
      model.add(Activation('relu'))
      model.add(Dense(self.action_space))
      model.add(Activation('softmax'))    
      model.compile(optimizer = Adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
      return model       

    
    def save_weight_bias(self,now):
	#print self.model_c1.summary()
        #weights = []

        weight1 = self.model.layers[0].get_weights()[0]	
        biase1 = self.model.layers[0].get_weights()[1] #2,4
        
        weight2 = self.model.layers[2].get_weights()[0]
        biase2 = self.model.layers[2].get_weights()[1] #2,4
        
        weight3 = self.model.layers[4].get_weights()[0]
        biase3 = self.model.layers[4].get_weights()[1] #2,4
        
        data = np.hstack([weight1, biase1, weight2, biase2, weight3, biase3])
        
        header = ["Weights between Input and Hidden1","Biases between Input and Hidden1",
          "Weights between Hidden1 and Hidden2","Biases between Hidden1 and Hidden2", 
          "Weights between Hidden2 and Output","Weights between Hidden2 and Output"]
        filename = "First_order/Weights/weights_biases_" + now.strftime("%Y%m%d-%H%M") + ".csv"

        with open(filename, 'w') as f:
            pd.DataFrame(rows = header).to_csv(f,encoding='utf-8', index=False, header = True)
            
        with open(filename, 'a') as f:
            pd.DataFrame(data).to_csv(f,encoding='utf-8', index=False, header = False)
    	

	
    def get_out_layers(self,state):
	#print self.model_c1.summary()
        funcs = None
        inp = self.model.input                                           # input placeholder
        inp = [inp]
        outputs = [layer.output for layer in self.model.layers]          # all layer outputs		
        funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
        		       		        	  	
        activations = []
        # Learning phase. 0 = Test mode (no dropout or batch normalization)
        layer_outputs = [func([state.reshape(1,len(state)), 0.])[0] for func in funcs]    	
        for layer_activations in layer_outputs:
            activations.append(layer_activations)		
        return activations

 
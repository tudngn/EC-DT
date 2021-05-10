# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:23:22 2019

@author: z5095790
"""


from sklearn.datasets import load_svmlight_file
from numpy import loadtxt
import numpy as np
import scipy.io


def read_data_txt():
   
 
    '''
    data = load_svmlight_file("NumericalData/txt-files/glass.scale.txt")
    x = data[0].todense()
    labels = np.reshape(data[1],(len(data[1]),1))
    for i in range(0,len(labels)):
        if labels[i] <= 3:
           labels[i] = labels[i]-1
        else:
           labels[i] = labels[i]-2'''
           
    '''       
    data = load_svmlight_file("NumericalData/txt-files/segment.scale.txt")
    x = data[0].todense()
    labels = np.reshape(data[1],(len(data[1]),1))
    
    for i in range(0,len(labels)):
        labels[i] = labels[i]-1'''
     
    '''    
    data = load_svmlight_file("NumericalData/txt-files/wine.scale.txt")
    x = data[0].todense()
    labels = np.reshape(data[1],(len(data[1]),1))
    
    for i in range(0,len(labels)):
        labels[i] = labels[i]-1'''
    
    '''    
    data = load_svmlight_file("NumericalData/txt-files/vehicle.scale.txt")
    x = data[0].todense()
    labels = np.reshape(data[1],(len(data[1]),1))
    
    for i in range(0,len(labels)):
        labels[i] = labels[i]-1'''
        
    '''    
    data = loadtxt("NumericalData/txt-files/data_banknote_authentication.txt", delimiter=",", unpack=False)'''
    
    '''
    mat = scipy.io.loadmat("NumericalData/txt-files/occupancy_data/data_all.mat")
    data = mat["data"]
    
    for i in range(0,data.shape[1]-1):
        max_value = np.max(data[:,i])
        min_value = np.min(data[:,i])
        data[:,i] = (data[:,i] - min_value)/max_value'''
    
    '''
    data = loadtxt("NumericalData/txt-files/wifi_localization.txt", delimiter="\t", unpack=False)
    data[:,-1] = data[:,-1] - 1
    for i in range(0,data.shape[1]-1):
        max_value = np.max(data[:,i])
        min_value = np.min(data[:,i])
        data[:,i] = (data[:,i] - min_value)/(max_value-min_value)
    
    return(data)'''
   
    data = load_svmlight_file("NumericalData/txt-files/ionosphere_scale.txt")
    x = data[0].todense()
    labels = np.reshape(data[1],(len(data[1]),1))
    x = x[:,2:np.shape(x)[1]]
    for i in range(0,x.shape[1]):
        max_value = np.max(x[:,i])
        min_value = np.min(x[:,i])
        x[:,i] = (x[:,i] - min_value)/(max_value-min_value)
    for i in range(0,len(labels)):
        if labels[i] == -1:
            labels[i] = 0

    return(np.array(np.hstack([x,labels])))

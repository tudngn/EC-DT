# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:23:22 2019

@author: z5095790
"""


import pandas as pd
import numpy as np
#import h5py
import scipy.io


def read_data_csv():
    
    '''
    df = pd.read_csv('NumericalData/data-files/magic04.data', header = None)
#    label = df.pop(10)
#    df.insert(0, column = 'label', value = label)
    d = df.values
    for i in range(0,len(d)):
        if d[i,-1] == 'g':
            d[i,-1] = 0               
        else:
            d[i,-1] = 1
    
    for i in range(0,d.shape[1]-1):
        max_value = np.max(d[:,i])
        min_value = np.min(d[:,i])
        d[:,i] = (d[:,i] - min_value)/(max_value-min_value)
    
    d = d[:,2:len(d)]'''
    
    '''
    df = pd.read_csv('NumericalData/data-files/wdbc.data', header = None)
#    label = df.pop(10)
#    df.insert(0, column = 'label', value = label)
    d = df.values
    d = np.delete(d, 0, 1)
    temp = np.copy(d[:,0])
    temp = np.reshape(temp,[len(temp),1])
    d = d[:,1:11]
    d = np.hstack([d,temp])
    for i in range(0,len(d)):
        if d[i,-1] == 'B':
            d[i,-1] = 0               
        else:
            d[i,-1] = 1 
            
    for i in range(0,d.shape[1]-1):
        max_value = np.max(d[:,i])
        min_value = np.min(d[:,i])
        d[:,i] = (d[:,i] - min_value)/(max_value-min_value)'''
    
    '''
    df = pd.read_csv('NumericalData/data-files/Wall-Following_Robot_Navigation/sensor_readings_2.data', header = None)
#    label = df.pop(10)
#    df.insert(0, column = 'label', value = label)
    d = df.values
    for i in range(0,len(d)):
        if d[i,-1] == 'Move-Forward':
            d[i,-1] = 0               
        elif d[i,-1] == 'Slight-Right-Turn':
            d[i,-1] = 1
        elif d[i,-1] == 'Sharp-Right-Turn':
            d[i,-1] = 2
        else:
            d[i,-1] = 3'''
    
    '''        
    df = pd.read_csv('NumericalData/new/data-files/balance-scale.data', header = None)
    d = df.values
    temp = np.copy(d[:,0])
    temp = np.reshape(temp,[len(temp),1])
    d = np.delete(d, 0, 1)    
    d = np.hstack([d,temp])
    for i in range(0,len(d)):
        if d[i,-1] == 'B':
            d[i,-1] = 0               
        elif d[i,-1] == 'L':
            d[i,-1] = 1
        else:
            d[i,-1] = 2
            
    #for i in range(0,d.shape[1]-1):
    #    max_value = np.max(d[:,i])
    #    min_value = np.min(d[:,i])
    #    d[:,i] = (d[:,i] - min_value)/(max_value-min_value)'''
    
    '''
    df = pd.read_csv('NumericalData/new/data-files/dermatology.data', header = None)
    d = df.values
    d = np.delete(d,[10,33],1)
    d[:,-1] = d[:,-1] - 1
    #d = d.astype(float)
    for i in range(0,d.shape[1]-1):
        max_value = np.max(d[:,i])
        min_value = np.min(d[:,i])
        d[:,i] = (d[:,i] - min_value)/(max_value-min_value)'''
     
    '''    
    mat = scipy.io.loadmat("NumericalData/new/data-files/popfailures.mat")
    d = mat["popfailures"]
    d = np.delete(d,[11,12,13],1)
    return(np.array(d))'''
    
    '''
    mat = scipy.io.loadmat("NumericalData/new/data-files/page-blocks.mat")
    d = mat["pageblocks"]
    d[:,-1] = d[:,-1] - 1
    
    for i in range(0,d.shape[1]-1):
        max_value = np.max(d[:,i])
        min_value = np.min(d[:,i])
        d[:,i] = (d[:,i] - min_value)/(max_value-min_value)'''
        
        
    df = pd.read_csv('NumericalData/new/data-files/sonar.data', header = None)
    d = df.values
    for i in range(0,len(d)):
        if d[i,-1] == 'R':
            d[i,-1] = 0               
        else:
            d[i,-1] = 1
    
    
    return(d.astype(float))       


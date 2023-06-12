# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Recurrent Neural netowrk
#Part 1 - Data Preprocessing


#importing the libraries
import numpy as np#
import matplotlib.pyplot as plt
import pandas as pd
#importing the training set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2 ].values
#getting the first index 
#makes a numpy array of 1 column(google stock price)

#Part 2 - Building the RNN


#Part 3 - Making predictions and visualising results
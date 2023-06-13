# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

#Feature Scaling(using standardisation or normalisation)
#to help improve performance
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled =  sc.fit_transform(training_set)#used to copute min and max values to be used for scaling

#creating a data structure with 60 timesteps and 1 output
#each time t the rnn is going to look at the 60 stock prices(days) before time t and try to predict next

#Initialize empty lists for training data
X_train = []
y_train = []
# Loop over the scaled training set
for i in range(60, 1258):
    # For each index i, add the sequence of the previous 60 data points to X_train
    X_train.append(training_set_scaled[i-60:i, 0])
    # Add the current data point to y_train
    y_train.append(training_set_scaled[i, 0])

# Convert lists to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping to 3d array, to allow RNN to rocess and understand dynamics
#the three parameters basically here are(1d-stock prices, 2d - timesteps, 3d - number of indicators )
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

#PART 2 - Building the RNN(stacked lstm architechture)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN

regressor = Sequential()

#Adding first LSTM layer and some Dropout Regularisation







#PART 3 - Making the predictions and visualise the results
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
#first layer(true for tracking)
regressor.add(LSTM(units = 50, return_sequences=True, input_shape = (X_train.shape[1],1) ))

regressor.add(Dropout(0.2))
#Dropout is a regularization technique to prevent overfitting
#Adding second LSTM layer and some Dropout Regularisation
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
#Adding third LSTM layer and some Dropout Regularisation
regressor.add(LSTM(units = 50, return_sequences=True ))
regressor.add(Dropout(0.2))
#Adding fourth LSTM layer and some Dropout Regularisation
regressor.add(LSTM(units = 50, return_sequences=False ))
regressor.add(Dropout(0.2))

#adding output layer
regressor.add(Dense(units=1))

#compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fitting the RNN to the training set
#regressor.fit(X_train, y_train,epochs= 100, batch_size = 32)
#PART 3 - Making the predictions and visualise the results


#getting the real stock price of jan 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#getting the predicted stock price of jan 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
#visualising the results
X_test = []
for i in range(60, 80):
    # For each index i, add the sequence of the previous 60 data points to X_train
    X_test.append(inputs[i-60:i, 0])
   
# Convert lists to numpy arrays
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#Visualise the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')

plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()
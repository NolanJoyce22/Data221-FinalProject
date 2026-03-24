
import pandas as pd

#   ***     File Processing     ***
#read file
dataset=pd.read_csv("student-mat.csv")
#select features
selectedFeatures=['Fjob', 'Mjob', 'Medu', 'Fedu', 'address', 'traveltime','famsup', 'famsize',  'internet', 'Pstatus', 'schoolsup']

#get features (x) and target label (y)
x=dataset[selectedFeatures]
y=dataset['G3']

#check if dataset has missing values
print(x.isnull().sum())


#Encoding categorical data to binary
categoricalColumns=['address', 'famsize', 'Pstatus','Mjob','Fjob', 'famsup', 'internet', 'schoolsup']
xEncoded=pd.get_dummies(x, columns=categoricalColumns, drop_first=True)

#verify encoded data has no missing values
print(xEncoded.isnull().sum())

from sklearn.model_selection import train_test_split
#make train-test split
    #Set test split to 0.20, consistent with other ML models
    #random state=42 to ensure shuffling and equal train test split
features_train, features_test, labels_train, labels_test=train_test_split(xEncoded, y,test_size=0.20, random_state=42)

#normalize features
    #normalizing features is key for neural networks to ensure optimization and equal treatment of features
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
features_train=scaler.fit_transform(features_train)
features_test=scaler.transform(features_test)

import tensorflow as tf
tf.random.set_seed(42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

#set model to sequential
neural_network_model = Sequential()

#create and add input layer
#one neuron per feature variable (11)
input_layer = InputLayer(input_shape=(xEncoded.shape[1],))
neural_network_model.add(input_layer)

#create and add first hidden layer
    #relu activation to ensure optimization
hidden_layer = Dense(64, activation='relu')
neural_network_model.add(hidden_layer)

#create and add second hidden layer
    #relu activation to ensure optimization
second_hidden_layer = Dense(32, activation='relu')
neural_network_model.add(second_hidden_layer)

#create and add third hidden layer
    #relu activation to ensure optimization
third_hidden_layer = Dense(16, activation='relu')
neural_network_model.add(third_hidden_layer)

#create and add output layer
output_layer = Dense(1)
neural_network_model.add(output_layer)

#compile model setting loss to mse, metrics to mae
    #adam optimizer
neural_network_model.compile(loss='mse', metrics=['mae'], optimizer='adam')

#fit model to training set, epochs to 30 to ensure consistent data
neural_network_model.fit(features_train, labels_train, epochs=30, validation_split=0.2)

#get predictions
predictions = neural_network_model.predict(features_test)

import numpy as np

#evaluate mse, rmse on testing set
loss, mae = neural_network_model.evaluate(features_test, labels_test)
rmse = np.sqrt(loss)

#print results to user
print(f"RMSE: {rmse}")
print(f"MSE: {loss}")
print(f"MAE: {mae}")


import pandas as pd

#read file
dataset=pd.read_csv("student-mat.csv")
#set feature vs target features
x=dataset[['Fjob', 'Mjob', 'Medu', 'Fedu', 'address', 'traveltime','famsup', 'famsize',  'internet', 'Pstatus', 'schoolsup']]
y=dataset['G3']

#check if dataset has missing values
print(x.isnull().sum())


#Encoding categorical data to binary
x["address"]=x["address"].map({"U":1, "R":0})
x["famsize"]=x["famsize"].map({"LE3":1, "GT3":0})
x["Pstatus"]=x["Pstatus"].map({"T":1, "A":0})
x["Mjob"]=x["Mjob"].map({"teacher":1, "health":2, "services":3, "at_home":4, "other":5})
x["Fjob"] = x["Fjob"].map({"teacher":1, "health":2, "services":3, "at_home":4, "other":5})
x["famsup"]=x["famsup"].map({"yes":1, "no":0})
x["internet"]=x["internet"].map({"yes":1, "no":0})
x["schoolsup"]=x["schoolsup"].map({"yes":1, "no":0})


print(x.isnull().sum())

from sklearn.model_selection import train_test_split
#make train-test split
features_train, features_test, labels_train, labels_test=train_test_split(x, y,test_size=0.20)

import tensorflow as tf
#TODO
#decide what is a good seed number
tf.random.set_seed(1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

#set model to sequential
neural_network_model = Sequential()


#create and add input layer
#one neuron per feature variable (11)
input_layer = InputLayer(input_shape=(x.shape[1],))
neural_network_model.add(input_layer)

#create and add first hidden layer
hidden_layer = Dense(3, activation='relu')
neural_network_model.add(hidden_layer)

#create and add second hidden layer
second_hidden_layer = Dense(5, activation='relu')
neural_network_model.add(second_hidden_layer)

#create and add third hidden layer
third_hidden_layer = Dense(2, activation='relu')
neural_network_model.add(third_hidden_layer)

#TODO
#what kind of activation do we want

#create and add output layer
output_layer = Dense(1)
neural_network_model.add(output_layer)

#TODO
#optimizer?
neural_network_model.compile(loss='mse')

#TODO
#decide number of epochs (around 30 should be ok for this dataset)
neural_network_model.fit(features_train, labels_train, epochs=10)
neural_network_model.evaluate(features_test, labels_test)

#get predictions
predictions = neural_network_model.predict(features_test)

import numpy as np

#evaluate mse, rmse
mse = neural_network_model.evaluate(features_test, labels_test)
rmse = np.sqrt(mse)

#print results to user
print(f"rmse: {rmse}")
print(f"mse: {mse}")


import pandas as pd

#   ***     File Processing     ***
#read file
dataset=pd.read_csv("student-mat.csv")
#select features
selectedFeatures=['Fjob', 'Mjob', 'Medu', 'Fedu', 'address', 'traveltime',
                  'famsup', 'famsize',  'internet', 'Pstatus', 'schoolsup']

#get features (x) and target label (y)
x=dataset[selectedFeatures]
y=dataset['G3']

#check if dataset has missing values
print(x.isnull().sum())


#Encoding categorical data to binary
categoricalColumns=['address', 'famsize', 'Pstatus','Mjob','Fjob',
                    'famsup', 'internet', 'schoolsup']
xEncoded=pd.get_dummies(x, columns=categoricalColumns, drop_first=True)

#verify encoded data has no missing values
print(xEncoded.isnull().sum())


#   ***     Train-Test Split     ***

from sklearn.model_selection import train_test_split
#make train-test split
    #Set test split to 0.20, consistent with other ML models
    #random state=42 to ensure shuffling and equal train test split
features_train, features_test, labels_train, labels_test=(train_test_split(
    xEncoded, y,test_size=0.20, random_state=42))

#normalize features
    #normalizing features is key for neural networks to ensure optimization and equal treatment of features
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
features_train=scaler.fit_transform(features_train)
features_test=scaler.transform(features_test)


#   ***     Neural Network Model Implementation     ***

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

#create and add forth hidden layer
    #relu activation to ensure optimization
fourth_hidden_layer = Dense(8, activation='relu')
neural_network_model.add(fourth_hidden_layer)

#create and add output layer
output_layer = Dense(1)
neural_network_model.add(output_layer)

#compile model setting loss to mse, metrics to mae
    #adam optimizer
neural_network_model.compile(loss='mse', metrics=['mae'], optimizer='adam')

#fit model to training set, epochs to 30 to ensure consistent data
neural_network_model.fit(features_train, labels_train, epochs=30, validation_split=0.2)


#   ***     Model Interpretation     ***


#get predictions
predictions = neural_network_model.predict(features_test)

import numpy as np
from sklearn.metrics import r2_score

#evaluate mse, rmse on testing set
loss, mae = neural_network_model.evaluate(features_test, labels_test)
rmse = np.sqrt(loss)
r2= r2_score(labels_test, predictions.flatten())

#print results to user
print(f"\nRMSE: {rmse}")
print(f"MSE: {loss}")
print(f"MAE: {mae}")
print(f"R-SQUARED: {r2}")


#   ***     Model Visualizations     ***

import matplotlib.pyplot as plt1

#flatten predictions
predictions_flat = predictions.flatten()


# #Scatter plot to represent predicted labels vs actual labels
plt1.scatter(labels_test, predictions, color='navy')

#set min and max graph values
min_val = min(labels_test.min(), predictions_flat.min())
max_val = max(labels_test.max(), predictions_flat.max())

#plot a perfect prediction line
plt1.plot([min_val, max_val],
          [min_val, max_val], color='hotpink',
          linestyle='--', label='Perfect Prediction')

#labels and title for scatter plot
plt1.xlabel("Actual Values (G3)")
plt1.ylabel("Predicted Values")
plt1.title("Actual vs Predicted Grades (Neural Network Model)")

plt1.legend()
plt1.show()


#Residual plot to visualize error in predictions
import matplotlib.pyplot as plt2

#compute residuals
residuals = labels_test - predictions_flat
#create scatter plot where x=predicted values, y=residuals
plt2.scatter(labels_test, residuals, color='navy')

#plot line at y=0 to represent perfect predictions (no error)
plt2.axhline(y=0, color='hotpink',linestyle='--')

#labels and titles for residual plot
plt2.ylabel("Residuals (actual - predicted)")
plt2.xlabel("Predicted Values")
plt2.title("Residual Plot (Neural Network Model)")
plt2.show()


#Loss graph to represent model fit

import matplotlib.pyplot as plt3

#assign variable to model fit
history = neural_network_model.fit(features_train, labels_train, epochs=30, validation_split=0.2)
#plot training loss over each epoch
plt3.plot(history.history['loss'], label='Training Loss', color='navy')
#plot validation loss over each epoch
plt3.plot(history.history['val_loss'], label='Validation Loss', color='hotpink')

#labels and titles for loss plot
plt3.xlabel("Epochs")
plt3.ylabel("Loss (MSE)")
plt3.title("Training vs Validation Loss")
plt3.legend()
plt3.show()

#get model summary
neural_network_model.summary()

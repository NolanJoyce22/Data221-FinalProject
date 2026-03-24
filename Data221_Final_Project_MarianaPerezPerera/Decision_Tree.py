#importing the needed packages:
from pyexpat import model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np

#loading in the dataset:
student_alcohol_df = pd.read_csv('student-mat.csv')

#selecting the propper/key features we are using:
selected_features = [
    #the parents background:
    'Medu', 'Fedu', 'Mjob', 'Fjob',
    #the living conditions:
    'address', 'Pstatus', 'traveltime', 'internet',
    #the family demographics:
    'famsup', 'famsize', 'famrel'
]

#creating the target variable which is G3:
target_variable = 'G3'

#now we have make sure we are not including everything else (removing the leakage)
x = student_alcohol_df[selected_features]
y= student_alcohol_df[target_variable]

#converting the categorical values into numerical:
#converting address: (U-values=1 and R-values=0)
x["address"] = x["address"].map({"U":1, "R":0})
#famsize: (LE3=1 and GT3=0)
x["famsize"] = x["famsize"].map({"LE3":1, "GT3":0})
#Pstatus: (T=1, A=0)
x["Pstatus"] = x["Pstatus"].map({"T":1, "A": 0})
#famsup: (yes=1 and no=0)
x["famsup"] = x["famsup"].map({"yes":1, "no":0})
#internet: (yes=1 and no=0)
x["internet"] = x["internet"].map({"yes":1, "no":0})
#doing Mjob and Fjob together because they have multiple different categorical answers instead of just 2:
x = pd.get_dummies(x, columns=["Mjob", "Fjob"], drop_first=True)


#train/test splitting into 80/20
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#creating the controlled model:
decision_tree_regressor = DecisionTreeRegressor(
    #moderate depth for best results:
    max_depth=4,
    #greater than one to help generalization:
    min_samples_leaf = 5,
    min_samples_split=10,
    random_state=42
)

#training the model:
decision_tree_regressor.fit(x_train, y_train)

#evaluating the model:
#predictions on test and train:
#train:
train_predictions = decision_tree_regressor.predict(x_train)
#test:
test_predictions = decision_tree_regressor.predict(x_test)

#mean_squared_error on both:
#Train:
train_mse = mean_squared_error(y_train, train_predictions)
train_rmse = np.sqrt(train_mse)

#Test:
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)


#train:
train_r2 = r2_score(y_train, train_predictions)
#test:
test_r2 = r2_score(y_test, test_predictions)

#printing each one:
#training and testing RMSE:
print("The training mean squared error is ", train_rmse)
print("The test mean squared error is ", test_rmse)
#training and test R squared:
print("\nThe training r-squared is ", train_r2)
print("The test r-squared is ", test_r2)

#feature importance:
importance = pd.Series(
    decision_tree_regressor.feature_importances_,
    index=x.columns
).sort_values(ascending=False)

#printing the importance:
print("\nThe feature importance:\n", importance)
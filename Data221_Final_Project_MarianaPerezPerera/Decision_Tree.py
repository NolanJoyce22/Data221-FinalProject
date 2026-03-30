#importing the needed packages:
from pyexpat import model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.tree import plot_tree
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

#loading in the dataset:
student_alcohol_dataFrame = pd.read_csv(r'C:\Users\maria\Documents\School\Data221\Data 221-Final Project\student-mat.csv')

#selecting the propper/key features we are using and creating the feature matrix:
selected_features = [
    #the parents background:
    'Medu', 'Fedu', 'Mjob', 'Fjob',
    #the living conditions:
    'address', 'Pstatus', 'traveltime', 'internet',
    #the family demographics:
    'famsup', 'famsize', 'famrel'
]

#creating the target variable which is G3:
target_variable ='G3'

# #now we have make sure we are not including everything else (removing the leakage)
x = student_alcohol_dataFrame[selected_features]
y= student_alcohol_dataFrame[target_variable]

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

#R-squared:
#train:
train_r2 = r2_score(y_train, train_predictions)
#test:
test_r2 = r2_score(y_test, test_predictions)

#Mean absolute error on testing data:
test_mae = mean_absolute_error(y_test, test_predictions)

#printing each one (for the testing data only):
#squared error:
print("Mean squared error (MSE): ", test_mse)
#root MSE: (will penalize big errors)
print("Root mean squared error (RMSE): ", test_rmse)
#average error size:
print("Mean absolute error (MAE): ", test_mae)
#the overall performance:
print("R²: ", test_r2)

#The model overall is worse at predicting the average grade

#feature importance:
importance = pd.Series(
    decision_tree_regressor.feature_importances_,
    index=x.columns
).sort_values(ascending=False)

#printing the importance:
print("\nThe feature importance:\n", importance)


#----------------------------------------------------------------
#Visuals:
#Plotting the importance features on a bar graph:
importance.plot(kind= 'bar', color="pink")
plt.title("Decision Tree Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.xticks(rotation= 45)
plt.show()

#Plotting the predicted vs actual results on a scatter plot:
plt.scatter(y_test, test_predictions, color='pink')
plt.xlabel('Actual Values (G3)')
plt.ylabel('Predicted Values (G3)')
plt.title("The Actual vs Predicted Grades (Decision Tree)")
#plotting the line of perfection:
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='green', linestyle='--')
plt.show()

#creating a residual plot:
residuals = y_test - test_predictions
plt.scatter(test_predictions, residuals, color='pink')
plt.axhline(y=0, color='green')
plt.title("Residual Plot (Decision Tree)")
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Actual - Predicted)')

#Creating a Decision Tree Visualization:
plt.figure(figsize = (25,15))
plot_tree(decision_tree_regressor, feature_names=x.columns, filled=True,
          rounded=True, max_depth=2, fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()
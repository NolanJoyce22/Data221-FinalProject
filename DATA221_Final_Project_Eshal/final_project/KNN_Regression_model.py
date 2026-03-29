import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#loading dataset
df= pd.read_csv("student-mat.csv")

#------------------------------------------
#dropping the leakage features
#------------------------------------------

#removing two columns from dataframe: G1 and G2 (these are earlier grades and highly correlated with G3)
#   since keeping them would cause data leakage and inflate model preformance
df=df.drop(columns=["G1", "G2"])

#--------------------------------------------
#Feature selection
#--------------------------------------------

#Select only the features relevent to our research question
#features added are absences, studytime and schoolsup to make the features more 'Academic'
selected_features= ["Medu", "Fedu", "Mjob", "Fjob",
                    "address", "traveltime", "studytime","failures","absences", "internet",
                    'famsup', "schoolsup", "famrel"]

#split features x, and target y
x=df[selected_features] #input variables (predictors)
y=df["G3"] #target variable (final grade)

#-------------------------------------------
#Encoding categorical variables
#----------------------------------------------


#Indentifying categorical columns manually
#i removed Pstatus and famsize and instead added schoolsup to make the model more lean
categorical_cols= ['Mjob', 'Fjob', 'address', "famsup", "internet", "schoolsup"]

#Apply one-hot encoding:
#   this converts categorical values into binary (0/1) columns
# example- Mjob=teacher -> Mjob_teacher=1, others=0
#drop_first=True removes one category per feature to avoid redundancy
x_encoded= pd.get_dummies(x, columns=categorical_cols, drop_first=True)


#-------------------------------------
#Train-Test split
#--------------------------------------

#split the data into training (80%) and testing (20%) sets
#random_state ensures reproducibility (same split every run)
x_train, x_test, y_train, y_test= train_test_split(
    x_encoded, y, test_size=0.2, random_state=42
)

#-------------------------------------------------
#Scale all features (very important for KNN)
#------------------------------------------------

#initializing the scaler
scaler= StandardScaler()

#fit the scaler only on the training data and transform it
#prevents data leakage from the test set
x_train=scaler.fit_transform(x_train)

#apply the same transformation onto the test set
#(don't fit again on test data)
x_test=scaler.transform(x_test)

#----------------------------------------------
#train KNN model
#---------------------------------------------

#increased the number of neighbors
knn= KNeighborsRegressor(n_neighbors=15)
knn.fit(x_train,y_train)

#--------------------------------------------
#make predictions
#---------------------------------------------
y_pred=knn.predict(x_test)

#-------------------------------------------------
#evaluate model
#------------------------------------------------
mae=mean_absolute_error(y_test, y_pred)
mse=mean_squared_error(y_test, y_pred)
rmse= np.sqrt(mse)
r2= r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

print("Sample Predictions:", y_pred[:5])

#-------------------Scatter plot------------------------
#shows the error (actual-predicted) for each guess
residuals=y_test-y_pred

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.6, color='purple')

#horizontal line at 0 indicating no error
plt.axhline(y=0, color='black', linestyle='-')

plt.xlabel("Predicted Values (G3)")
plt.ylabel("Residuals (Actual- Predicted)")
plt.title("Residual Plot (KNN Regression Model")
plt.show()





#------------Graph comparison-------------------

plt.figure(figsize=(8,6))
#creating scatter plot
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')

#creating the diagonal 'perfect prediction' line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='--')

#labels and title
plt.xlabel("Actual Values (G3)")
plt.ylabel("Predicted Values (G3)")
plt.title("Actual vs Predicted (KNN Regression Model)")
plt.show()


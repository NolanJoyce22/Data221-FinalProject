import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load data set
student_alcohol_data = pd.read_csv("student-mat.csv")
# Create feature matrix
x = student_alcohol_data[["address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "traveltime", "famsup",
                          "internet", "famrel"]]
# Create target vector
y = student_alcohol_data["G3"]

# Map features with categorical feature to a numerical variable
x["address"] = x["address"].map({"U":1, "R":0})
x["famsize"] = x["famsize"].map({"LE3":1, "GT3":0})
x["Pstatus"] = x["Pstatus"].map({"T":1, "A":0})
x["famsup"] = x["famsup"].map({"yes":1, "no":0})
x["internet"] = x["internet"].map({"yes":1, "no":0})
# One-Hot code features with more than two categorical variables
x = pd.get_dummies(x, columns=["Mjob", "Fjob"], drop_first=True)

# Create train-test-split for model with 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
# Initialize scaler to standardize features
scaler = StandardScaler()
# fit the scaler on the training data only to avoid data leakage(mean = 0, std = 1)
x_train_scaled = scaler.fit_transform(x_train)
# Use the same scaling parameters to transform the test features
x_test_scaled = scaler.transform(x_test)

# Initialize LinearRegression model set fit_intercept to True
model = LinearRegression(fit_intercept=True)
# Train the model
model.fit(x_train_scaled, y_train)
# Predict the target labels
y_pred = model.predict(x_test_scaled)

print(y_pred)
print("R²:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))






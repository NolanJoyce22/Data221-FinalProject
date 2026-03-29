import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Load data set
student_alcohol_data = pd.read_csv(r"C:\Data 221\Data221 Final Project Nolan Joyce\student-mat.csv")
# Create feature matrix
x = student_alcohol_data[["address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "traveltime", "famsup",
                          "internet", "famrel"]]
# Create target vector
y = student_alcohol_data["G3"]
# convert categorical data into numerical columns so that the features are not treated as having order or ranking
#
x = pd.get_dummies(x, drop_first=True)

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
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)


#---------------------------------------------------------------
# Visual


# Scatter plot: Actual Vs Predicted
plt.figure()
plt.scatter(y_test, y_pred, alpha = 0.6)

# Perfect prediction line (y = x)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--")

# Labels and title
plt.xlabel("Actual Values (G3)")
plt.ylabel("Predicted Values (G3)")
plt.title("actual vs Predicted (Linear Regression)")

plt.show()

# Low actual values predictions are too high.
# High actual values predictions are too low.
# Regression towards the mean.
# Model struggles to capture extreme outcomes.




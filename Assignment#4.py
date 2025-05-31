# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
dataset = pd.read_csv('E:\AICLASS\Python Projects\insurance.csv')

# Step 2: Extract the relevant columns
age = dataset['age']
bmi = dataset['bmi']
charges = dataset['charges']

# Step 3: Analyze the relationship between 'age' and 'bmi'

# Scatter plot for visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=age, y=bmi)
plt.title('Scatter Plot: Age vs BMI')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.show()

# Linear Regression Model for Age vs BMI
X_bmi = age.values.reshape(-1, 1)  # Reshaping age as a 2D array for model fitting
y_bmi = bmi

# Create a linear regression model
regressor_bmi = LinearRegression()
regressor_bmi.fit(X_bmi, y_bmi)

# Predicting values using the model
y_bmi_pred = regressor_bmi.predict(X_bmi)

# Plot the regression line
plt.figure(figsize=(8, 6))
sns.scatterplot(x=age, y=bmi, label='Data Points')
plt.plot(age, y_bmi_pred, color='red', label='Regression Line')
plt.title('Linear Regression: Age vs BMI')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.legend()
plt.show()

# Print regression details
print(f"Linear regression equation for Age vs BMI: y = {regressor_bmi.intercept_:.2f} + {regressor_bmi.coef_[0]:.2f}*Age")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_bmi, y_bmi_pred):.2f}")
print(f"R-squared: {r2_score(y_bmi, y_bmi_pred):.2f}")

# Step 4: Analyze the relationship between 'age' and 'charges'

# Scatter plot for visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=age, y=charges)
plt.title('Scatter Plot: Age vs Charges')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()

# Linear Regression Model for Age vs Charges
X_charges = age.values.reshape(-1, 1)  # Reshaping age as a 2D array for model fitting
y_charges = charges

# Create a linear regression model
regressor_charges = LinearRegression()
regressor_charges.fit(X_charges, y_charges)

# Predicting values using the model
y_charges_pred = regressor_charges.predict(X_charges)

# Plot the regression line
plt.figure(figsize=(8, 6))
sns.scatterplot(x=age, y=charges, label='Data Points')
plt.plot(age, y_charges_pred, color='red', label='Regression Line')
plt.title('Linear Regression: Age vs Charges')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend()
plt.show()

# Print regression details
print(f"Linear regression equation for Age vs Charges: y = {regressor_charges.intercept_:.2f} + {regressor_charges.coef_[0]:.2f}*Age")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_charges, y_charges_pred):.2f}")
print(f"R-squared: {r2_score(y_charges, y_charges_pred):.2f}")

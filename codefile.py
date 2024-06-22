import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')

# Step 1: Reading and Understanding the Data
df = pd.read_csv("Bengaluru_House_Data.csv")

# Step 2: Data Cleaning and Preparation
# Handling missing values
df = df.dropna(subset=['total_sqft', 'price'])
df['size'] = df['size'].fillna(df['size'].mode()[0])
df['location'] = df['location'].fillna(df['location'].mode()[0])
df['society'] = df['society'].fillna(df['society'].mode()[0])
df['bath'] = df['bath'].fillna(df['bath'].mean())
df['balcony'] = df['balcony'].fillna(df['balcony'].mean())

# Encoding categorical variables
label_encoder = LabelEncoder()
df["availability"] = label_encoder.fit_transform(df["availability"])
df["area_type"] = label_encoder.fit_transform(df["area_type"])
df["society"] = label_encoder.fit_transform(df["society"])
df["size"] = label_encoder.fit_transform(df["size"])
df["location"] = label_encoder.fit_transform(df["location"])

# Step 3: Visualizing the Data
# Example visualizations are provided as per your original code, omitted for brevity.

# Step 4: Dummy Variables (if needed)
# Example not explicitly needed for your task.

# Step 5: Deriving New Features (if needed)
# Example not explicitly needed for your task.

# Step 7: Train-Test Split and Feature Scaling
X = df.drop(columns=['balcony'])
Y = df['balcony']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

# Step 8: Model Building
# Decision Tree Regressor
dt_model = DecisionTreeRegressor()
dt_model.fit(x_train.values, y_train.values)
print("Decision Tree Accuracy:", dt_model.score(x_test, y_test)*100)

# Random Forest Regressor
rf_model = RandomForestRegressor()
rf_model.fit(x_train.values, y_train.values)
print("Random Forest Accuracy:", rf_model.score(x_test, y_test)*100)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
print("Linear Regression Accuracy:", lr_model.score(x_test, y_test)*100)

# Lasso Regression
lasso_model = Lasso()
lasso_model.fit(x_train, y_train)
y_pred_lasso = lasso_model.predict(x_test)
print("Lasso Regression Mean Squared Error:", mean_squared_error(y_test, y_pred_lasso))

# Ridge Regression
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(x_train, y_train)
y_pred_ridge = ridge_model.predict(x_test)
print("Ridge Regression Mean Squared Error:", mean_squared_error(y_test, y_pred_ridge))

# Step 9: Prediction and Evaluation
# Example metrics and evaluation are provided as per your original code, omitted for brevity.

# Step 10: Save Model
filename = 'saved_model.pkl'
try:
    with open(filename, 'wb') as file:
        pickle.dump(rf_model, file)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving the model: {e}")

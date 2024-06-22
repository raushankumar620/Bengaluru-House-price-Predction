<h2> Problem Statement & Approach Summary</h2>

We aim to predict the number of balconies (balcony) in properties using various machine learning algorithms such as Decision Tree, Random Forest, Linear Regression, Lasso Regression, and Ridge Regression. This involves preprocessing data, exploring correlations, visualizing insights, and training models to achieve accurate predictions.

Steps and Tasks Summary
Reading and Understanding the Data

Import necessary libraries (pandas, matplotlib, seaborn, numpy, sklearn).
Read the dataset (Bengaluru_House_Data.csv) into a pandas DataFrame (df).
Data Cleaning and Preparation

Handle missing values using appropriate methods (fillna, dropping rows).
Encode categorical variables using LabelEncoder.
Visualizing the Data

Explore data distribution and relationships using visualizations like histograms, box plots, and heatmaps.
Analyze correlations between features.
Feature Engineering

Create dummy variables for categorical features if needed.
Derive new features or transform existing ones as necessary.
Model Building

Split the dataset into training and testing sets.
Train various regression models (Decision Tree, Random Forest, Linear Regression, Lasso Regression, Ridge Regression) on the training data.
Prediction and Evaluation

Evaluate model performance using metrics like R-squared and Mean Squared Error (MSE).
Save the best-performing model using pickle.

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 16:08:35 2023

@author: USER
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Recreate the dataset
np.random.seed(42)

# Generate data for 50 existing customers
customer_ids = range(1, 51)
avg_monthly_spend = np.random.uniform(100, 1000, 50)
purchase_frequency = np.random.uniform(1, 30, 50)
avg_purchase_value = np.random.uniform(50, 200, 50)
customer_lifetime = np.random.uniform(1, 60, 50)

# Calculate CLV as a product of average monthly spend and customer lifetime
clv = avg_monthly_spend * customer_lifetime

# Create a DataFrame for existing customers
existing_customers = pd.DataFrame({
    'Customer ID': customer_ids,
    'Average Monthly Spend': avg_monthly_spend,
    'Purchase Frequency': purchase_frequency,
    'Average Purchase Value': avg_purchase_value,
    'Customer Lifetime (months)': customer_lifetime,
    'CLV': clv
})

# Prepare the data
X = existing_customers[['Average Monthly Spend', 'Purchase Frequency', 'Average Purchase Value', 'Customer Lifetime (months)']]
y = existing_customers['CLV']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Evaluate the model
y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save the model as a pickle file
model_pickle_path = 'clv_lr_model.pkl'
with open(model_pickle_path, 'wb') as file:
    pickle.dump(lr_model, file)

mse, r2



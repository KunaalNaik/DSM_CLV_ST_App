# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 16:13:59 2023

@author: USER
"""
import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model_path = 'clv_lr_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

st.title('Customer Lifetime Value Predictor - it worked!')

st.sidebar.title('This is side bar!')

# Take user inputs
avg_monthly_spend = st.number_input('Average Monthly Spend', min_value=0.0, max_value=10000.0, step=0.1)
purchase_frequency = st.number_input('Purchase Frequency', min_value=0.0, max_value=100.0, step=0.1)
avg_purchase_value = st.number_input('Average Purchase Value', min_value=0.0, max_value=10000.0, step=0.1)
customer_lifetime = st.number_input('Customer Lifetime (months)', min_value=0, max_value=120, step=1)

# Create a dataframe from the inputs
input_data = pd.DataFrame([[avg_monthly_spend, purchase_frequency, avg_purchase_value, customer_lifetime]],
                          columns=['Average Monthly Spend', 'Purchase Frequency', 'Average Purchase Value', 'Customer Lifetime (months)'])

# Predict the CLV
if st.button('Predict CLV'):
    clv_prediction = model.predict(input_data)
    st.write(f'The predicted Customer Lifetime Value (CLV) is: {clv_prediction[0]:.2f}')


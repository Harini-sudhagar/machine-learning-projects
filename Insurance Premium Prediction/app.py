import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load trained model
with open("insurance_ridge_model.pkl", "rb") as f:
    model_data = pickle.load(f)

weights = model_data["weights"]
scaler = model_data["scaler"]
feature_columns = model_data["feature_columns"]


st.title("Insurance Premium Prediction")

st.write("Enter customer details to predict the insurance premium.")

# User inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

annual_income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0
)

dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=2
)

health_score = st.number_input(
    "Health Score",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

previous_claims = st.number_input(
    "Previous Claims",
    min_value=0,
    max_value=20,
    value=1
)

vehicle_age = st.number_input(
    "Vehicle Age",
    min_value=0.0,
    max_value=50.0,
    value=5.0
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300.0,
    max_value=850.0,
    value=700.0
)

insurance_duration = st.number_input(
    "Insurance Duration",
    min_value=1,
    max_value=20,
    value=5
)

# Categorical inputs
gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Married", "Single"]
)

education = st.selectbox(
    "Education Level",
    ["High School", "Bachelor's", "Master's", "PhD"]
)

occupation = st.selectbox(
    "Occupation",
    ["Employed", "Self-Employed", "Unemployed"]
)

location = st.selectbox(
    "Location",
    ["Urban", "Rural", "Suburban"]
)

policy_type = st.selectbox(
    "Policy Type",
    ["Basic", "Comprehensive", "Premium"]
)

customer_feedback = st.selectbox(
    "Customer Feedback",
    ["Poor", "Average", "Good"]
)

smoking_status = st.selectbox(
    "Smoking Status",
    ["Yes", "No"]
)

exercise_frequency = st.selectbox(
    "Exercise Frequency",
    ["Rarely", "Monthly", "Weekly", "Daily"]
)

property_type = st.selectbox(
    "Property Type",
    ["House", "Apartment", "Condo"]
)

# Date
policy_year = st.number_input(
    "Policy Start Year",
    min_value=2000,
    max_value=2030,
    value=2025
)

policy_month = st.number_input(
    "Policy Start Month",
    min_value=1,
    max_value=12,
    value=1
)

policy_day = st.number_input(
    "Policy Start Day",
    min_value=1,
    max_value=31,
    value=1
)


if st.button("Predict Premium"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Annual Income": [annual_income],
        "Number of Dependents": [dependents],
        "Health Score": [health_score],
        "Previous Claims": [previous_claims],
        "Vehicle Age": [vehicle_age],
        "Credit Score": [credit_score],
        "Insurance Duration": [insurance_duration],
        "Gender": [gender],
        "Marital Status": [marital_status],
        "Education Level": [education],
        "Occupation": [occupation],
        "Location": [location],
        "Policy Type": [policy_type],
        "Customer Feedback": [customer_feedback],
        "Smoking Status": [smoking_status],
        "Exercise Frequency": [exercise_frequency],
        "Property Type": [property_type],
        "Policy Start Year": [policy_year],
        "Policy Start Month": [policy_month],
        "Policy Start Day": [policy_day]
    })

    # One-hot encoding
    input_data = pd.get_dummies(
        input_data,
        drop_first=True,
        dtype=int
    )

    # Match training columns
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Add intercept
    input_b = np.c_[
        np.ones(input_scaled.shape[0]),
        input_scaled
    ]

    # Prediction
    prediction = input_b @ weights

    st.success(
        f"Predicted Premium Amount: ₹{prediction[0]:,.2f}"
    )
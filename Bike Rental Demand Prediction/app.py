import streamlit as st
import pandas as pd
import numpy as np


# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("hour.csv")


# -----------------------------
# Features and Target
# -----------------------------

features = ["temp", "atemp", "hum", "windspeed", "hr"]
target = "cnt"

X = df[features]
y = df[target]


# -----------------------------
# Train-Validation Split
# -----------------------------

np.random.seed(42)

indices = np.random.permutation(len(X))

train_end = int(0.60 * len(X))
val_end = int(0.80 * len(X))

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]


# -----------------------------
# Scaling
# -----------------------------

train_mean = X_train.mean()
train_std = X_train.std()

X_train_scaled = (
    X_train - train_mean
) / train_std


# -----------------------------
# Polynomial Features
# -----------------------------

def polynomial_features(X, degree):

    X_poly = X.copy()

    for d in range(2, degree + 1):
        X_poly = np.c_[
            X_poly,
            X ** d
        ]

    return X_poly


# -----------------------------
# Polynomial Degree
# -----------------------------

degree = 5


X_train_poly = polynomial_features(
    X_train_scaled.to_numpy(),
    degree
)


# -----------------------------
# Add Intercept
# -----------------------------

X_train_poly = np.c_[
    np.ones(X_train_poly.shape[0]),
    X_train_poly
]


# -----------------------------
# Normal Equation
# -----------------------------

weights = np.linalg.pinv(
    X_train_poly.T @ X_train_poly
) @ X_train_poly.T @ y_train.to_numpy()


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Bike Rental Demand Prediction")

st.write(
    "Predict the total number of bike rentals "
    "based on weather and time."
)


st.subheader("Enter Bike Rental Details")


temp = st.number_input(
    "Temperature (normalized)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)


atemp = st.number_input(
    "Feeling Temperature (normalized)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)


hum = st.number_input(
    "Humidity (normalized)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)


windspeed = st.number_input(
    "Windspeed (normalized)",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.01
)


hr = st.number_input(
    "Hour",
    min_value=0,
    max_value=23,
    value=12,
    step=1
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Bike Demand"):

    input_data = pd.DataFrame(
        [[
            temp,
            atemp,
            hum,
            windspeed,
            hr
        ]],
        columns=features
    )


    # Scale input using training statistics

    input_scaled = (
        input_data - train_mean
    ) / train_std


    # Polynomial transformation

    input_poly = polynomial_features(
        input_scaled.to_numpy(),
        degree
    )


    # Add intercept

    input_poly = np.c_[
        np.ones(input_poly.shape[0]),
        input_poly
    ]


    # Prediction

    prediction = input_poly @ weights


    # Bike rentals cannot be negative

    prediction = max(0, prediction[0])


    st.success(
        f"Predicted Bike Rentals: {prediction:.0f}"
    )
import streamlit as st
import pandas as pd
import numpy as np
import pickle


# =========================================================
# SIGMOID
# =========================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# =========================================================
# LOGISTIC REGRESSION CLASS
# =========================================================

class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.losses = []

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.losses = []

        for epoch in range(self.epochs):

            z = np.dot(X, self.weights) + self.bias

            predictions = sigmoid(z)

            error = predictions - y

            dw = np.dot(X.T, error) / len(X)

            db = np.mean(error)

            self.weights -= self.learning_rate * dw

            self.bias -= self.learning_rate * db

            loss = -np.mean(
                y * np.log(predictions + 1e-15)
                +
                (1 - y) * np.log(1 - predictions + 1e-15)
            )

            self.losses.append(loss)

    def predict_probability(self, X):

        X = np.asarray(X)

        z = np.dot(X, self.weights) + self.bias

        return sigmoid(z)

    def predict(self, X, threshold=0.5):

        probabilities = self.predict_probability(X)

        return (probabilities >= threshold).astype(int)


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Taxpayer Risk Classification",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD SAVED MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open(
        "taxpayer_risk_model.pkl",
        "rb"
    ) as file:

        model_data = pickle.load(file)

    return model_data


model_data = load_model()

model = model_data["model"]

mean = model_data["mean"]

std = model_data["std"]

features = model_data["features"]


# =========================================================
# TITLE
# =========================================================

st.title("📊 Taxpayer Risk Classification")

st.write(
    "Logistic Regression based classification of taxpayers "
    "into Low Risk and High Risk."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Taxpayer Details")


revenue = st.sidebar.number_input(
    "Revenue",
    min_value=0.0,
    value=100000.0
)

expenses = st.sidebar.number_input(
    "Expenses",
    min_value=0.0,
    value=50000.0
)

tax_liability = st.sidebar.number_input(
    "Tax Liability",
    min_value=0.0,
    value=10000.0
)

tax_paid = st.sidebar.number_input(
    "Tax Paid",
    min_value=0.0,
    value=8000.0
)

late_filings = st.sidebar.number_input(
    "Late Filings",
    min_value=0,
    value=0,
    step=1
)

compliance_violations = st.sidebar.number_input(
    "Compliance Violations",
    min_value=0,
    value=0,
    step=1
)

profit = st.sidebar.number_input(
    "Profit",
    value=50000.0
)

tax_compliance_ratio = st.sidebar.number_input(
    "Tax Compliance Ratio",
    min_value=0.0,
    value=0.8
)

audit_findings = st.sidebar.number_input(
    "Audit Findings",
    min_value=0,
    value=0,
    step=1
)


# =========================================================
# PREDICTION
# =========================================================

if st.sidebar.button("Predict Risk"):

    input_data = pd.DataFrame(
        [[
            revenue,
            expenses,
            tax_liability,
            tax_paid,
            late_filings,
            compliance_violations,
            profit,
            tax_compliance_ratio,
            audit_findings
        ]],
        columns=features
    )

    # Same scaling used during training
    input_scaled = (
        input_data - mean
    ) / std

    # Prediction probability
    probability = model.predict_probability(
        input_scaled.values
    )[0]

    # 0.5 threshold
    if probability >= 0.5:
        risk = "High Risk"
    else:
        risk = "Low Risk"

    # =====================================================
    # RESULT
    # =====================================================

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    col1.metric(
        "High Risk Probability",
        f"{probability * 100:.2f}%"
    )

    col2.metric(
        "Classification",
        risk
    )

    if risk == "High Risk":

        st.error("⚠️ High Risk")

    else:

        st.success("✅ Low Risk")


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Algorithm",
    "Logistic Regression"
)

col2.metric(
    "Features",
    len(features)
)

col3.metric(
    "Threshold",
    "0.50"
)


# =========================================================
# FEATURES
# =========================================================

st.subheader("Features Used")

st.write(features)
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="University Admission Prediction",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# LOAD SAVED MODEL
# =========================================================

with open("university_admission_model.pkl", "rb") as file:
    model_data = pickle.load(file)

weights = model_data["weights"]
bias = model_data["bias"]
scaler = model_data["scaler"]
features = model_data["features"]


# =========================================================
# FUNCTIONS
# =========================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_logistic_regression(
    X_train,
    y_train,
    learning_rate=0.01,
    iterations=1000
):

    n_samples, n_features = X_train.shape

    weights_temp = np.zeros(n_features)
    bias_temp = 0

    losses = []

    for i in range(iterations):

        # Linear equation
        z = np.dot(X_train, weights_temp) + bias_temp

        # Probability
        predictions = sigmoid(z)

        # Avoid log(0)
        predictions = np.clip(
            predictions,
            1e-15,
            1 - 1e-15
        )

        # Log Loss
        loss = -np.mean(
            y_train * np.log(predictions)
            +
            (1 - y_train) * np.log(1 - predictions)
        )

        losses.append(loss)

        # Gradient
        dw = (
            1 / n_samples
        ) * np.dot(
            X_train.T,
            predictions - y_train
        )

        db = (
            1 / n_samples
        ) * np.sum(
            predictions - y_train
        )

        # Update weights and bias
        weights_temp -= learning_rate * dw
        bias_temp -= learning_rate * db

    return weights_temp, bias_temp, losses


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("Admission_Predict.csv")

    df.columns = df.columns.str.strip()

    return df


df = load_data()


# =========================================================
# CREATE TARGET
# =========================================================

df["admission"] = (
    df["Chance of Admit"] >= 0.5
).astype(int)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

df["average_test_score"] = (
    df["GRE Score"] + df["TOEFL Score"]
) / 2


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎓 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Student Prediction",
        "Model Performance",
        "Dataset Analysis",
        "Learning Rate Comparison",
        "Feature Importance",
        "Model Information"
    ]
)


# =========================================================
# PAGE 1 — STUDENT PREDICTION
# =========================================================

if page == "Student Prediction":

    st.title("🎓 University Admission Prediction")

    st.write(
        "Predict a student's admission probability "
        "using Logistic Regression."
    )

    st.divider()

    st.subheader("Enter Student Details")

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # LEFT COLUMN
    # -----------------------------------------------------

    with col1:

        gre = st.number_input(
            "GRE Score",
            min_value=260,
            max_value=340,
            value=300
        )

        toefl = st.number_input(
            "TOEFL Score",
            min_value=0,
            max_value=120,
            value=100
        )

        university_rating = st.number_input(
            "University Rating",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=1.0
        )

        sop = st.number_input(
            "SOP",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5
        )

    # -----------------------------------------------------
    # RIGHT COLUMN
    # -----------------------------------------------------

    with col2:

        lor = st.number_input(
            "LOR",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5
        )

        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=8.0,
            step=0.01
        )

        research = st.selectbox(
            "Research Experience",
            [0, 1]
        )

    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    average_test_score = (
        gre + toefl
    ) / 2

    st.info(
        f"Average Test Score: {average_test_score:.2f}"
    )

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if st.button(
        "🔮 Predict Admission",
        type="primary"
    ):

        input_data = pd.DataFrame({

            "GRE Score": [gre],

            "TOEFL Score": [toefl],

            "University Rating": [
                university_rating
            ],

            "SOP": [sop],

            "LOR": [lor],

            "CGPA": [cgpa],

            "Research": [research],

            "average_test_score": [
                average_test_score
            ]
        })

        # Same feature order
        input_data = input_data[features]

        # Scale
        input_scaled = scaler.transform(
            input_data
        )

        # Logistic Regression equation
        z = np.dot(
            input_scaled,
            weights
        ) + bias

        # Probability
        probability = sigmoid(z)[0]

        # Class prediction
        prediction = int(
            probability >= 0.5
        )

        st.divider()

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Admission Probability",
                f"{probability * 100:.2f}%"
            )

        with col2:

            if prediction == 1:

                st.success(
                    "🎓 Prediction: Admitted"
                )

            else:

                st.warning(
                    "Prediction: Not Admitted"
                )

        st.progress(
            float(probability)
        )


# =========================================================
# PAGE 2 — MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Evaluation of the trained Logistic Regression model."
    )

    # -----------------------------------------------------
    # PREPARE DATA
    # -----------------------------------------------------

    X = df[features]

    y = df["admission"]

    # Scale
    X_scaled = scaler.transform(X)

    # Probability
    z = np.dot(
        X_scaled,
        weights
    ) + bias

    y_probability = sigmoid(z)

    # Prediction
    y_prediction = (
        y_probability >= 0.5
    ).astype(int)

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y,
        y_prediction
    )

    precision = precision_score(
        y,
        y_prediction,
        zero_division=0
    )

    recall = recall_score(
        y,
        y_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y,
        y_prediction,
        zero_division=0
    )

    auc_score = roc_auc_score(
        y,
        y_probability
    )

    # -----------------------------------------------------
    # DISPLAY METRICS
    # -----------------------------------------------------

    st.subheader("Performance Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

    col2.metric(
        "Precision",
        f"{precision:.2%}"
    )

    col3.metric(
        "Recall",
        f"{recall:.2%}"
    )

    col4.metric(
        "F1 Score",
        f"{f1:.2%}"
    )

    col5.metric(
        "AUC",
        f"{auc_score:.3f}"
    )

    st.divider()

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    st.subheader("🔲 Confusion Matrix")

    cm = confusion_matrix(
        y,
        y_prediction
    )

    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "Actual Label"
    )

    ax.set_title(
        "Confusion Matrix"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Not Admitted", "Admitted"]
    )

    ax.set_yticklabels(
        ["Not Admitted", "Admitted"]
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(fig)

    st.divider()

    # -----------------------------------------------------
    # ROC CURVE
    # -----------------------------------------------------

    st.subheader("📈 ROC Curve")

    fpr, tpr, thresholds = roc_curve(
        y,
        y_probability
    )

    fig, ax = plt.subplots()

    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {auc_score:.3f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve"
    )

    ax.legend()

    st.pyplot(fig)


# =========================================================
# PAGE 3 — DATASET ANALYSIS
# =========================================================

elif page == "Dataset Analysis":

    st.title("📂 Dataset Analysis")

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Admitted",
        int(df["admission"].sum())
    )

    col4.metric(
        "Not Admitted",
        int(
            (df["admission"] == 0).sum()
        )
    )

    st.divider()

    # -----------------------------------------------------
    # DATASET PREVIEW
    # -----------------------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------------
    # MISSING VALUES
    # -----------------------------------------------------

    st.subheader("Missing Values")

    missing_values = df.isnull().sum()

    missing_df = pd.DataFrame({

        "Column": missing_values.index,

        "Missing Values": missing_values.values

    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------------
    # CLASS DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "Admission Class Distribution"
    )

    class_counts = (
        df["admission"]
        .value_counts()
        .sort_index()
    )

    class_counts.index = [
        "Not Admitted",
        "Admitted"
    ]

    st.bar_chart(
        class_counts
    )


# =========================================================
# PAGE 4 — LEARNING RATE COMPARISON
# =========================================================

elif page == "Learning Rate Comparison":

    st.title("⚙️ Learning Rate Comparison")

    st.write(
        "Comparison of different learning rates "
        "for Logistic Regression trained from scratch."
    )

    st.info(
        "Lower Log Loss indicates better training performance."
    )

    # -----------------------------------------------------
    # PREPARE DATA
    # -----------------------------------------------------

    X = df[features]

    y = df["admission"]

    # -----------------------------------------------------
    # SAME 60/20/20 SPLIT AS NOTEBOOK
    # -----------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # -----------------------------------------------------
    # SCALE USING TRAINING DATA
    # -----------------------------------------------------

    from sklearn.preprocessing import StandardScaler

    comparison_scaler = StandardScaler()

    X_train_scaled = comparison_scaler.fit_transform(
        X_train
    )

    X_val_scaled = comparison_scaler.transform(
        X_val
    )

    # -----------------------------------------------------
    # LEARNING RATES
    # -----------------------------------------------------

    learning_rates = [
        0.001,
        0.01,
        0.1,
        1.0
    ]

    results = []

    all_losses = {}

    # -----------------------------------------------------
    # TRAIN EACH LEARNING RATE
    # -----------------------------------------------------

    for lr in learning_rates:

        temp_weights, temp_bias, losses = (
            train_logistic_regression(
                X_train_scaled,
                y_train.to_numpy(),
                learning_rate=lr,
                iterations=1000
            )
        )

        final_loss = losses[-1]

        results.append({

            "Learning Rate": lr,

            "Final Log Loss": final_loss

        })

        all_losses[lr] = losses

    # -----------------------------------------------------
    # RESULTS TABLE
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df[
        "Final Log Loss"
    ] = results_df[
        "Final Log Loss"
    ].round(6)

    st.subheader(
        "Learning Rate Results"
    )

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # -----------------------------------------------------
    # LOSS COMPARISON GRAPH
    # -----------------------------------------------------

    st.subheader(
        "📉 Loss Convergence"
    )

    fig, ax = plt.subplots()

    for lr in learning_rates:

        ax.plot(
            all_losses[lr],
            label=f"LR = {lr}"
        )

    ax.set_xlabel(
        "Iterations"
    )

    ax.set_ylabel(
        "Log Loss"
    )

    ax.set_title(
        "Learning Rate vs Log Loss"
    )

    ax.legend()

    st.pyplot(fig)

    st.divider()

    # -----------------------------------------------------
    # FINAL MODEL CONVERGENCE
    # -----------------------------------------------------

    st.subheader(
        "Final Model Convergence"
    )

    final_losses = all_losses[0.01]

    fig, ax = plt.subplots()

    ax.plot(
        final_losses
    )

    ax.set_xlabel(
        "Iterations"
    )

    ax.set_ylabel(
        "Log Loss"
    )

    ax.set_title(
        "Log Loss Convergence - Learning Rate 0.01"
    )

    st.pyplot(fig)

    st.success(
        "Learning Rate 0.01 is used as the final model "
        "configuration in this project."
    )


# =========================================================
# PAGE 5 — FEATURE IMPORTANCE
# =========================================================

elif page == "Feature Importance":

    st.title("🔍 Feature Importance")

    coefficient_df = pd.DataFrame({

        "Feature": features,

        "Coefficient": weights

    })

    coefficient_df[
        "Absolute Coefficient"
    ] = np.abs(
        coefficient_df["Coefficient"]
    )

    coefficient_df = (
        coefficient_df
        .sort_values(
            "Absolute Coefficient",
            ascending=False
        )
    )

    st.subheader(
        "Feature Coefficients"
    )

    st.dataframe(
        coefficient_df,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "Top Features"
    )

    top_features = (
        coefficient_df
        .head(10)
        .set_index("Feature")
        ["Absolute Coefficient"]
    )

    st.bar_chart(
        top_features
    )

    st.info(
        "Larger absolute coefficient values indicate "
        "stronger influence in the Logistic Regression model. "
        "This does not mean the feature directly causes admission."
    )


# =========================================================
# PAGE 6 — MODEL INFORMATION
# =========================================================

elif page == "Model Information":

    st.title("ℹ️ Model Information")

    st.subheader(
        "Machine Learning Algorithm"
    )

    st.write(
        "Logistic Regression implemented from scratch "
        "using NumPy."
    )

    st.subheader(
        "Model Components"
    )

    st.write("""
    • Sigmoid Function

    • Log Loss

    • Gradient Descent

    • Learning Rate

    • Learned Weights

    • Bias

    • StandardScaler
    """)

    st.subheader(
        "Saved Model"
    )

    st.write(
        "The trained model parameters and scaler are "
        "loaded from university_admission_model.pkl."
    )

    st.subheader(
        "Features Used"
    )

    for feature in features:

        st.write(
            f"• {feature}"
        )
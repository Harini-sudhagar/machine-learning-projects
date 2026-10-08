import streamlit as st
import numpy as np
import pandas as pd
import random
import time
import re
import joblib
from collections import deque


# ============================================================
# 1. MULTINOMIAL NAIVE BAYES
#    Same class as main.ipynb
# ============================================================

class MultinomialNaiveBayes:

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):

        self.classes = np.unique(y)

        self.class_log_prior = {}
        self.feature_log_prob = {}

        for category in self.classes:

            class_mask = (y == category)

            X_class = X[class_mask]

            class_count = X_class.shape[0]

            self.class_log_prior[category] = np.log(
                class_count / len(y)
            )

            word_counts = X_class.sum(axis=0)

            total_words = word_counts.sum()

            probabilities = (
                word_counts + self.alpha
            ) / (
                total_words + self.alpha * X.shape[1]
            )

            self.feature_log_prob[category] = np.log(
                probabilities
            )

    def predict(self, X):

        predictions = []

        for document in X:

            class_scores = {}

            for category in self.classes:

                score = self.class_log_prior[category]

                score += np.sum(
                    document *
                    self.feature_log_prob[category]
                )

                class_scores[category] = score

            prediction = max(
                class_scores,
                key=class_scores.get
            )

            predictions.append(prediction)

        return np.array(predictions)

    def predict_proba(self, X):

        probabilities = []

        for document in X:

            class_scores = []

            for category in self.classes:

                score = self.class_log_prior[category]

                score += np.sum(
                    document *
                    self.feature_log_prob[category]
                )

                class_scores.append(score)

            class_scores = np.array(
                class_scores
            )

            class_scores -= np.max(
                class_scores
            )

            exp_scores = np.exp(
                class_scores
            )

            probs = (
                exp_scores /
                exp_scores.sum()
            )

            probabilities.append(probs)

        return np.array(probabilities)


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Live Survey Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_data = joblib.load(
        "sentiment_model.joblib"
    )

    model = model_data["model"]

    word_to_index = model_data[
        "word_to_index"
    ]

    vocabulary_list = model_data[
        "vocabulary_list"
    ]

    return (
        model,
        word_to_index,
        vocabulary_list
    )


best_model, word_to_index, vocabulary_list = load_model()


# ============================================================
# 4. STOP WORDS
#    Same as main.ipynb
# ============================================================

stop_words = {
    "the", "is", "a", "an", "and", "or",
    "of", "to", "in", "on", "for",
    "with", "this", "that", "it",
    "as", "are", "was", "were",
    "be", "by", "from", "at",
    "which", "but", "have",
    "has", "had", "they",
    "you", "he", "she",
    "we", "i", "their",
    "our", "your", "my"
}


# ============================================================
# 5. NEGATORS
#    Same as main.ipynb
# ============================================================

negators = {
    "not",
    "no",
    "never",
    "cannot",
    "hardly",
    "without"
}


# ============================================================
# 6. PREPROCESSING
#    Same as main.ipynb
# ============================================================

def preprocess_text(text):

    text = text.lower()

    words = re.findall(
        r"\b[a-z]+\b",
        text
    )

    processed_words = []

    negation = False

    for word in words:

        if word in negators:

            negation = True

            processed_words.append(
                word
            )

            continue

        if word in stop_words:

            continue

        if negation:

            processed_words.append(
                "NOT_" + word
            )

            negation = False

        else:

            processed_words.append(
                word
            )

    return processed_words


# ============================================================
# 7. BAG OF WORDS
#    Same as main.ipynb
# ============================================================

def create_bow(
    tokens,
    word_to_index
):

    vector = np.zeros(
        len(word_to_index),
        dtype=int
    )

    for word in tokens:

        if word in word_to_index:

            vector[
                word_to_index[word]
            ] += 1

    return vector


# ============================================================
# 8. LIVE SURVEY RESPONSES
#    Based on the same project topics
# ============================================================

positive_responses = [

    "payroll is fast and easy to use",

    "I love how simple accounting is",

    "invoicing saved us hours every week",

    "great experience with customer support",

    "mobile app works perfectly",

    "very happy with reports",

    "time tracking was quick to set up",

    "excellent experience with account setup",

    "project tracking is intuitive and useful",

    "I have no complaints about expense reimbursement"

]


negative_responses = [

    "payroll is confusing and slow",

    "very disappointed with accounting",

    "invoicing keeps crashing",

    "I cannot figure out customer support",

    "mobile app is too expensive",

    "terrible experience with reports",

    "time tracking does not work",

    "account setup is frustrating",

    "project tracking was a waste of time",

    "expense reimbursement is unreliable"

]


neutral_responses = [

    "payroll is okay",

    "accounting does what it says",

    "average experience with invoicing",

    "I have used time tracking a few times",

    "customer support is fine",

    "no strong opinion about reports",

    "mobile app is acceptable for now",

    "account setup works as described",

    "project tracking is standard",

    "expense reimbursement is neither good nor bad"

]


# ============================================================
# 9. GENERATE LIVE RESPONSE
# ============================================================

def generate_live_response(
    response_number
):

    # Simulated negative incident
    # between response 50 and 79

    if 50 <= response_number < 80:

        sentiment = random.choices(
            [
                "negative",
                "neutral",
                "positive"
            ],
            weights=[
                0.85,
                0.10,
                0.05
            ]
        )[0]

    else:

        sentiment = random.choices(
            [
                "negative",
                "neutral",
                "positive"
            ],
            weights=[
                0.20,
                0.30,
                0.50
            ]
        )[0]

    if sentiment == "positive":

        return random.choice(
            positive_responses
        )

    elif sentiment == "negative":

        return random.choice(
            negative_responses
        )

    else:

        return random.choice(
            neutral_responses
        )


# ============================================================
# 10. PREDICT SENTIMENT
# ============================================================

def predict_live_response(text):

    tokens = preprocess_text(
        text
    )

    bow = create_bow(
        tokens,
        word_to_index
    )

    bow = bow.reshape(
        1,
        -1
    )

    prediction = best_model.predict(
        bow
    )[0]

    probabilities = best_model.predict_proba(
        bow
    )[0]

    probability_dict = {
        category: probability
        for category, probability
        in zip(
            best_model.classes,
            probabilities
        )
    }

    return (
        prediction,
        probability_dict
    )


# ============================================================
# 11. SESSION STATE
# ============================================================

if "running" not in st.session_state:

    st.session_state.running = False


if "response_number" not in st.session_state:

    st.session_state.response_number = 0


if "results" not in st.session_state:

    st.session_state.results = []


if "recent_predictions" not in st.session_state:

    st.session_state.recent_predictions = deque(
        maxlen=50
    )


# ============================================================
# 12. TITLE
# ============================================================

st.title(
    "📊 Live Survey Sentiment Analyzer"
)

st.write(
    "Real-Time Survey Sentiment Analysis "
    "using Multinomial Naive Bayes"
)


# ============================================================
# 13. SIDEBAR
# ============================================================

st.sidebar.header(
    "Controls"
)


delay = st.sidebar.slider(
    "Response Delay (seconds)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5
)


alert_threshold = st.sidebar.slider(
    "Negative Alert Threshold",
    min_value=0.20,
    max_value=0.80,
    value=0.40,
    step=0.05
)


# ============================================================
# 14. MODEL INFORMATION
# ============================================================

st.sidebar.markdown(
    "---"
)

st.sidebar.write(
    f"**Vocabulary Size:** "
    f"{len(vocabulary_list)}"
)

st.sidebar.write(
    f"**Best Alpha:** "
    f"{best_model.alpha}"
)


# ============================================================
# 15. BUTTONS
# ============================================================

col1, col2, col3 = st.columns(
    3
)


with col1:

    if st.button(
        "▶ Start",
        use_container_width=True
    ):

        st.session_state.running = True


with col2:

    if st.button(
        "⏹ Stop",
        use_container_width=True
    ):

        st.session_state.running = False


with col3:

    if st.button(
        "🔄 Reset",
        use_container_width=True
    ):

        st.session_state.running = False

        st.session_state.response_number = 0

        st.session_state.results = []

        st.session_state.recent_predictions = deque(
            maxlen=50
        )

        st.rerun()


# ============================================================
# 16. MANUAL SURVEY INPUT
# ============================================================

st.subheader(
    "📝 Test a Survey Response"
)

manual_text = st.text_input(
    "Enter a survey response"
)

if st.button(
    "🔍 Analyze Response"
):

    if manual_text.strip():

        prediction, probabilities = (
            predict_live_response(
                manual_text
            )
        )

        st.write(
            f"**Predicted Sentiment:** "
            f"{prediction.upper()}"
        )

        probability_df = pd.DataFrame(
            {
                "Sentiment": [
                    "Positive",
                    "Neutral",
                    "Negative"
                ],
                "Probability": [
                    probabilities.get(
                        "positive",
                        0
                    ),
                    probabilities.get(
                        "neutral",
                        0
                    ),
                    probabilities.get(
                        "negative",
                        0
                    )
                ]
            }
        )

        st.bar_chart(
            probability_df.set_index(
                "Sentiment"
            )
        )

    else:

        st.warning(
            "Please enter a survey response."
        )


# ============================================================
# 17. PLACEHOLDERS
# ============================================================

incoming_placeholder = st.empty()

metrics_placeholder = st.empty()

chart_placeholder = st.empty()

table_placeholder = st.empty()

alert_placeholder = st.empty()


# ============================================================
# 18. LIVE SIMULATION
# ============================================================

if st.session_state.running:

    response_number = (
        st.session_state.response_number
        + 1
    )

    response = generate_live_response(
        response_number
    )

    prediction, probabilities = (
        predict_live_response(
            response
        )
    )

    st.session_state.response_number = (
        response_number
    )

    negative_probability = (
        probabilities.get(
            "negative",
            0
        )
    )

    st.session_state.recent_predictions.append(
        prediction
    )

    st.session_state.results.append(
        {

            "Response":
                response_number,

            "Survey Response":
                response,

            "Prediction":
                prediction,

            "Negative Probability":
                round(
                    negative_probability,
                    4
                ),

            "Positive Probability":
                round(
                    probabilities.get(
                        "positive",
                        0
                    ),
                    4
                ),

            "Neutral Probability":
                round(
                    probabilities.get(
                        "neutral",
                        0
                    ),
                    4
                )
        }
    )


    # ========================================================
    # INCOMING RESPONSE
    # ========================================================

    with incoming_placeholder.container():

        st.subheader(
            f"📥 Incoming Survey Response "
            f"#{response_number}"
        )

        st.info(
            response
        )

        st.write(
            f"**Predicted Sentiment:** "
            f"{prediction.upper()}"
        )


        # Current probabilities

        probability_display = pd.DataFrame(
            {
                "Sentiment": [
                    "Positive",
                    "Neutral",
                    "Negative"
                ],
                "Probability": [
                    probabilities.get(
                        "positive",
                        0
                    ),
                    probabilities.get(
                        "neutral",
                        0
                    ),
                    probabilities.get(
                        "negative",
                        0
                    )
                ]
            }
        )

        st.bar_chart(
            probability_display.set_index(
                "Sentiment"
            )
        )


    # ========================================================
    # METRICS
    # ========================================================

    total_responses = len(
        st.session_state.results
    )

    positive_count = sum(
        1
        for r in st.session_state.results
        if r["Prediction"] == "positive"
    )

    neutral_count = sum(
        1
        for r in st.session_state.results
        if r["Prediction"] == "neutral"
    )

    negative_count = sum(
        1
        for r in st.session_state.results
        if r["Prediction"] == "negative"
    )


    negative_share = (
        negative_count /
        total_responses
        if total_responses > 0
        else 0
    )


    with metrics_placeholder.container():

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Total Responses",
            total_responses
        )

        col2.metric(
            "Positive",
            positive_count
        )

        col3.metric(
            "Neutral",
            neutral_count
        )

        col4.metric(
            "Negative",
            negative_count
        )


    # ========================================================
    # NEGATIVE ALERT
    # ========================================================

    if negative_share >= alert_threshold:

        alert_placeholder.error(
            f"🚨 ALERT: Negative sentiment is "
            f"{negative_share:.1%}, exceeding the "
            f"{alert_threshold:.1%} threshold."
        )

    else:

        alert_placeholder.success(
            f"Sentiment status normal — "
            f"Negative share: "
            f"{negative_share:.1%}"
        )


    # ========================================================
    # CHART
    # ========================================================

    chart_data = pd.DataFrame(
        st.session_state.results
    )

    chart_data = chart_data[
        [
            "Response",
            "Negative Probability",
            "Positive Probability",
            "Neutral Probability"
        ]
    ]

    chart_data = chart_data.set_index(
        "Response"
    )


    with chart_placeholder.container():

        st.subheader(
            "📈 Sentiment Probability Over Time"
        )

        st.line_chart(
            chart_data
        )


    # ========================================================
    # RECENT RESULTS TABLE
    # ========================================================

    recent_data = pd.DataFrame(
        st.session_state.results[-10:]
    )


    with table_placeholder.container():

        st.subheader(
            "📝 Recent Survey Responses"
        )

        st.dataframe(
            recent_data,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # NEXT RESPONSE
    # ========================================================

    time.sleep(
        delay
    )

    st.rerun()


# ============================================================
# 19. INITIAL / STOPPED STATE
# ============================================================

else:

    if len(
        st.session_state.results
    ) == 0:

        st.info(
            "Click ▶ Start to begin the "
            "live survey simulation."
        )

    else:

        st.subheader(
            "📊 Final Results"
        )

        final_data = pd.DataFrame(
            st.session_state.results
        )

        st.dataframe(
            final_data,
            use_container_width=True,
            hide_index=True
        )
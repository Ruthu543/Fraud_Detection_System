
import os
import joblib
import pandas as pd
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🏦",
    layout="wide"
)

# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():

    xgb_model = joblib.load(
        os.path.join(MODEL_DIR, "xgb_model.pkl")
    )

    rf_model = joblib.load(
        os.path.join(MODEL_DIR, "rf_model.pkl")
    )

    if_model = joblib.load(
        os.path.join(MODEL_DIR, "if_model.pkl")
    )

    supervised_scaler = joblib.load(
        os.path.join(MODEL_DIR, "supervised_scaler.pkl")
    )

    if_scaler = joblib.load(
        os.path.join(MODEL_DIR, "if_scaler.pkl")
    )

    supervised_features = joblib.load(
        os.path.join(MODEL_DIR, "supervised_feature_columns.pkl")
    )

    isolation_features = joblib.load(
        os.path.join(MODEL_DIR, "isolation_feature_columns.pkl")
    )

    return (
        xgb_model,
        rf_model,
        if_model,
        supervised_scaler,
        if_scaler,
        supervised_features,
        isolation_features
    )


try:
    (
        xgb_model,
        rf_model,
        if_model,
        supervised_scaler,
        if_scaler,
        supervised_features,
        isolation_features
    ) = load_models()

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
}

[data-testid="stMetric"] {
    background-color: #1e293b !important;
    padding: 20px !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

[data-testid="stMetricDelta"] {
    color: #22c55e !important;
}

.stMarkdown, .stText {
    color: inherit;
}

div.stButton > button {
    background-color: #1455a0;
    color: white;
    border-radius: 8px;
    height: 48px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #0d3d75;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🏦 FraudGuard")

    st.caption("Credit Card Fraud Detection System")

    st.divider()

    st.success("All ML models loaded")

    st.write("🟢 XGBoost")
    st.write("🟢 Random Forest")
    st.write("🟢 Isolation Forest")

    st.divider()

    st.caption("Machine Learning Powered Security")


# ==========================================
# HEADER
# ==========================================

st.title("Credit Card Fraud Detection")

st.markdown(
    "Analyze individual transactions using trained "
    "Machine Learning models."
)

st.divider()


# ==========================================
# MODEL INFORMATION
# ==========================================

col1, col2, col3 = st.columns(3)

col1.metric("ML Models", "3")
col2.metric("Features", "30")
col3.metric("Detection Mode", "Individual")


st.divider()


# ==========================================
# TRANSACTION INPUT FORM
# ==========================================

st.subheader("💳 Transaction Details")

st.info(
    "Enter all 30 transaction features. "
    "V1–V28 are PCA-transformed features from the "
    "original Credit Card Fraud dataset."
)

with st.form("transaction_form"):

    st.markdown("### Basic Transaction Information")

    col1, col2 = st.columns(2)

    with col1:
        time_value = st.number_input(
            "Transaction Time",
            min_value=0.0,
            value=0.0,
            help="Time in seconds from the dataset reference point."
        )

    with col2:
        amount_value = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=0.0,
            help="Transaction amount in the dataset currency."
        )

    st.divider()

    st.markdown("### PCA Features (V1–V28)")

    st.caption(
        "Enter the transformed feature values from your dataset."
    )

    feature_values = {}

    for start in range(1, 29, 3):

        cols = st.columns(3)

        for i, col in enumerate(cols):

            feature_number = start + i

            if feature_number <= 28:

                with col:

                    feature_values[f"V{feature_number}"] = (
                        st.number_input(
                            f"V{feature_number}",
                            value=0.0,
                            format="%.6f",
                            key=f"v{feature_number}"
                        )
                    )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Analyze Transaction",
        use_container_width=True
    )


# ==========================================
# PREDICTION
# ==========================================

if submitted:

    try:

        # Build input using saved feature order

        input_data = {
            "Time": time_value,
            "Amount": amount_value,
            **feature_values
        }

        raw_df = pd.DataFrame([input_data])

        # --------------------------------------
        # SUPERVISED MODEL PREPROCESSING
        # --------------------------------------

        X_supervised = raw_df[supervised_features].copy()

        X_supervised[["Time", "Amount"]] = (
            supervised_scaler.transform(
                X_supervised[["Time", "Amount"]]
            )
        )

        # --------------------------------------
        # ISOLATION FOREST PREPROCESSING
        # --------------------------------------

        X_if = raw_df[isolation_features].copy()

        X_if_scaled = if_scaler.transform(X_if)

        # --------------------------------------
        # XGBOOST PREDICTION
        # --------------------------------------

        xgb_probability = xgb_model.predict_proba(
            X_supervised
        )[0][list(xgb_model.classes_).index(1)]

        xgb_prediction = int(xgb_probability >= 0.80)

        # --------------------------------------
        # RANDOM FOREST PREDICTION
        # --------------------------------------

        rf_probability = rf_model.predict_proba(
            X_supervised
        )[0][list(rf_model.classes_).index(1)]

        rf_prediction = int(rf_probability >= 0.50)

        # --------------------------------------
        # ISOLATION FOREST PREDICTION
        # --------------------------------------

        if_raw_prediction = if_model.predict(
            X_if_scaled
        )[0]

        if_prediction = int(if_raw_prediction == -1)

        # --------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------

        st.divider()

        st.header("📊 Transaction Analysis Results")

        st.subheader("XGBoost Prediction")

        c1, c2 = st.columns(2)

        c1.metric(
            "Fraud Probability",
            f"{xgb_probability * 100:.2f}%"
        )

        c2.metric(
            "Decision Threshold",
            "80%"
        )

        if xgb_prediction == 1:
            st.error("🚨 XGBoost: FRAUD DETECTED")
        else:
            st.success("✅ XGBoost: NORMAL TRANSACTION")

        st.progress(float(xgb_probability))

        st.divider()

        st.subheader("Random Forest Prediction")

        c1, c2 = st.columns(2)

        c1.metric(
            "Fraud Probability",
            f"{rf_probability * 100:.2f}%"
        )

        c2.metric(
            "Decision Threshold",
            "50%"
        )

        if rf_prediction == 1:
            st.error("🚨 Random Forest: FRAUD DETECTED")
        else:
            st.success("✅ Random Forest: NORMAL TRANSACTION")

        st.progress(float(rf_probability))

        st.divider()

        st.subheader("Isolation Forest Prediction")

        if if_prediction == 1:
            st.warning("⚠️ ANOMALY DETECTED")
        else:
            st.success("✅ NORMAL PATTERN DETECTED")

        st.caption(
            "Isolation Forest detects unusual patterns. "
            "An anomaly is not necessarily a fraudulent transaction."
        )

        # --------------------------------------
        # COMPARISON TABLE
        # --------------------------------------

        st.divider()

        st.subheader("Model Comparison")

        results = pd.DataFrame({
            "Model": [
                "XGBoost",
                "Random Forest",
                "Isolation Forest"
            ],
            "Prediction": [
                "Fraud" if xgb_prediction else "Normal",
                "Fraud" if rf_prediction else "Normal",
                "Anomaly" if if_prediction else "Normal Pattern"
            ],
            "Fraud Probability": [
                f"{xgb_probability * 100:.2f}%",
                f"{rf_probability * 100:.2f}%",
                "Not available"
            ]
        })

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error(f"Prediction error: {e}")


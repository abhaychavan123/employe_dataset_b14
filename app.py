
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Employee Leave Prediction",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern layout, typography, and card shadows
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Top title styling */
    .title-banner {
        text-align: center;
        padding: 24px 10px;
        margin-bottom: 24px;
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    }
    .title-banner h1 {
        color: #0f172a;
        font-size: 2rem;
        margin-bottom: 8px;
    }
    .title-banner p {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 0;
    }

    /* Card styling with soft depth shadow */
    .custom-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04), 0 4px 6px -4px rgba(0, 0, 0, 0.02);
    }

    /* Prediction alert cards */
    .result-stay {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        border-radius: 16px;
        padding: 28px;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.4);
    }
    .result-leave {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        border-radius: 16px;
        padding: 28px;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 10px 20px -5px rgba(239, 68, 68, 0.4);
    }
    .result-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .result-subtitle {
        font-size: 0.95rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(path="navie_buyes.pkl"):
    """Load the pickled GaussianNB model."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


model = load_model("navie_buyes.pkl")

# Banner Section
st.markdown("""
<div class="title-banner">
    <h1>👔 Employee Retention Predictor</h1>
    <p>Predict whether an employee is likely to stay or leave using the trained Naive Bayes classifier</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file `navie_buyes.pkl` not found! Ensure the `.pkl` file is in the root directory.")
    st.stop()

# --- Inputs Form Layout ---
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📋 Academic & Location Profile")
    
    education_label = st.selectbox("Education Level", ["Bachelors", "Masters", "PhD"])
    # Standard label encoding matching common dataset splits
    education_map = {"Bachelors": 0, "Masters": 1, "PhD": 2}
    education = education_map[education_label]

    city_label = st.selectbox("City Location", ["Bangalore", "New Delhi", "Pune"])
    city_map = {"Bangalore": 0, "New Delhi": 1, "Pune": 2}
    city = city_map[city_label]

    joining_year = st.slider("Joining Year", min_value=2010, max_value=2024, value=2017, step=1)
    payment_tier = st.selectbox("Payment Tier", [1, 2, 3], index=2, 
                                help="Tier 1 (Highest payment grade) to Tier 3 (Base level)")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("👤 Personal & Experience Details")

    age = st.slider("Age", min_value=18, max_value=65, value=28, step=1)

    gender_label = st.radio("Gender", ["Female", "Male"], horizontal=True)
    gender_map = {"Female": 0, "Male": 1}
    gender = gender_map[gender_label]

    ever_benched_label = st.radio("Ever Put On Bench?", ["No", "Yes"], horizontal=True)
    ever_benched_map = {"No": 0, "Yes": 1}
    ever_benched = ever_benched_map[ever_benched_label]

    experience = st.slider("Experience In Current Domain (Years)", min_value=0, max_value=15, value=3, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

# Format into DataFrame with exact feature names matching training data
input_df = pd.DataFrame([{
    'Education': education,
    'JoiningYear': joining_year,
    'City': city,
    'PaymentTier': payment_tier,
    'Age': age,
    'Gender': gender,
    'EverBenched': ever_benched,
    'ExperienceInCurrentDomain': experience
}])

# --- Prediction Action Section ---
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.subheader("🔮 Run Inference")

pred_col1, pred_col2 = st.columns([1, 2])

with pred_col1:
    st.write("Evaluate model outcome on selected attributes:")
    run_prediction = st.button("🚀 Predict Employee Status", type="primary", use_container_width=True)

with pred_col2:
    if run_prediction:
        try:
            pred = model.predict(input_df)[0]
            probs = model.predict_proba(input_df)[0]

            if pred == 1:
                st.markdown(f"""
                <div class="result-leave">
                    <div class="result-title">⚠️ Likely to Leave (Class 1)</div>
                    <div class="result-subtitle">Model Confidence: <strong>{probs[1] * 100:.1f}%</strong> probability of churn</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-stay">
                    <div class="result-title">✅ Likely to Stay (Class 0)</div>
                    <div class="result-subtitle">Model Confidence: <strong>{probs[0] * 100:.1f}%</strong> probability of retention</div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.info("Click **Predict Employee Status** to generate the prediction.")

st.markdown('</div>', unsafe_allow_html=True)

with st.expander("🔍 View Raw Transformed Features"):
    st.dataframe(input_df, use_container_width=True)

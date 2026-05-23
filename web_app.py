import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set up page configuration
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Academic Performance Prediction")
st.write("Enter the student's details below to predict their final grade and academic status.")

# Load the pre-trained models from Part 5
@st.cache_resource
def load_models():
    with open('lin_reg_model.pkl', 'rb') as f:
        lin_model = pickle.load(f)
    with open('log_reg_model.pkl', 'rb') as f:
        log_model = pickle.load(f)
    return lin_model, log_model

try:
    lin_reg, log_reg = load_models()
except FileNotFoundError:
    st.error("⚠️ Model files (.pkl) not found! Make sure they are in the same folder as this script.")
    st.stop()

st.divider()

# --- INPUT FIELDS (Part 7 Requirements) ---
st.subheader("📊 Input Features")

col1, col2 = st.columns(2)

with col1:
    studytime_option = st.selectbox(
        "Study Time per Week",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "< 2 hours", 
            2: "2 - 5 hours", 
            3: "5 - 10 hours", 
            4: "> 10 hours"
        }[x]
    )
    
    failures = st.slider("Number of Past Class Failures", min_value=0, max_value=3, value=0)
    absences = st.number_input("Total Number of Absences", min_value=0, max_value=100, value=0, step=1)

with col2:
    g1 = st.slider("First Period Grade - G1 (0-20)", min_value=0, max_value=20, value=12)
    g2 = st.slider("Second Period Grade - G2 (0-20)", min_value=0, max_value=20, value=12)

# --- PROCESSING AND PREDICTION ---
if st.button("🔮 Predict Performance", type="primary"):
    # Structure the input data exactly like the training data layout
    user_data = pd.DataFrame([{
        'studytime': studytime_option,
        'failures': failures,
        'absences': absences,
        'G1': g1,
        'G2': g2
    }])
    
    # Generate predictions using Part 5 models
    predicted_g3_continuous = lin_reg.predict(user_data)[0]
    predicted_status_binary = log_reg.predict(user_data)[0]
    
    st.divider()
    st.subheader("🎯 Prediction Results")
    
    # Display Results in a clean dashboard card layout
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(
            label="Predicted Final Grade (Continuous G3)", 
            value=f"{predicted_g3_continuous:.2f} / 20"
        )
        
    with res_col2:
        status_text = "Pass" if predicted_status_binary == 1 else "Fail"
        color = "green" if status_text == "Pass" else "red"
        st.markdown(f"**Predicted Academic Status:** :{color}[{status_text}]")
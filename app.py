import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Load model and encoders
@st.cache_resource
def load_model():
    with open("model1.pkl", "rb") as f:
        model = pickle.load(f)
    with open("le_cpt.pkl", "rb") as f:
        le_cpt = pickle.load(f)
    with open("le_ins.pkl", "rb") as f:
        le_ins = pickle.load(f)
    with open("le_phys.pkl", "rb") as f:
        le_phys = pickle.load(f)
    return model, le_cpt, le_ins, le_phys

model, le_cpt, le_ins, le_phys = load_model()

st.title("🏥 CPT Denial Predictor")
st.write("Upload an Excel file with claim data to find the highest denied CPT code and its reason.")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Clean monetary columns
    df["Payment Amount"] = df["Payment Amount"].replace('[\$,]', '', regex=True).astype(float)
    df["Balance"] = df["Balance"].replace('[\$,]', '', regex=True).astype(float)

    # Label denial
    df["Denied"] = df["Denial Reason"].apply(lambda x: 0 if pd.isna(x) or x == "" else 1)

    # Encode categorical columns
    df["CPT Code Encoded"] = le_cpt.transform(df["CPT Code"])
    df["Insurance Encoded"] = le_ins.transform(df["Insurance Company"])
    df["Physician Encoded"] = le_phys.transform(df["Physician Name"])

    # Use the same feature columns as during training
    feature_cols = ["CPT Code Encoded", "Insurance Encoded", "Physician Encoded", "Payment Amount", "Balance"]
    X = df[feature_cols]

    # Make prediction
    df["Prediction"] = model.predict(X)

    # Filter predicted denials
    denied_df = df[df["Prediction"] == 1]

    if not denied_df.empty:
        # Group by CPT code
        summary = denied_df.groupby("CPT Code").agg({
            "Balance": "sum",
            "Denial Reason": lambda x: x.mode()[0] if not x.mode().empty else "N/A"
        }).reset_index()

        # Get highest denial
        top = summary.sort_values(by="Balance", ascending=False).head(1)

        st.subheader("📊 Highest Denied CPT Code")
        st.dataframe(top, use_container_width=True)
    else:
        st.success("No predicted denials in the uploaded file.")

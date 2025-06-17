import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
model = pickle.load(open('rf_model.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))

st.title(" Denial Code Prediction ")
st.markdown("Predict claim denial codes based on CPT, insurer, physician, and payment data.")

# Tabs: Single Entry | Batch Upload
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction"])

# --- Tab 1: Single Prediction ---
with tab1:
    st.subheader("Enter Claim Details")

    cpt_input = st.selectbox("CPT Code", encoders['CPT'].classes_)
    ins_input = st.selectbox("Insurance Company", encoders['Ins'].classes_)
    phys_input = st.selectbox("Physician Name", encoders['Phys'].classes_)
    payment = st.number_input("Payment Amount", min_value=0.0, format="%.2f")
    balance = st.number_input("Balance", min_value=0.0, format="%.2f")

    if st.button("🔮 Predict Denial Code"):
        try:
            input_df = pd.DataFrame([[
                encoders['CPT'].transform([cpt_input])[0],
                encoders['Ins'].transform([ins_input])[0],
                encoders['Phys'].transform([phys_input])[0],
                payment,
                balance
            ]], columns=['CPT Code', 'Insurance Company', 'Physician Name', 'Payment Amount', 'Balance'])

            pred = model.predict(input_df)[0]
            st.success(f"Predicted Denial Code: {int(pred)}")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Tab 2: Batch Prediction ---
with tab2:
    st.subheader("Upload Excel/CSV File")
    uploaded_file = st.file_uploader("Upload a file with CPT Code, Insurance Company, Physician Name, Payment Amount, Balance", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Encode using stored label encoders
            df['CPT Code'] = encoders['CPT'].transform(df['CPT Code'])
            df['Insurance Company'] = encoders['Ins'].transform(df['Insurance Company'])
            df['Physician Name'] = encoders['Phys'].transform(df['Physician Name'])

            # Predict
            preds = model.predict(df[['CPT Code', 'Insurance Company', 'Physician Name', 'Payment Amount', 'Balance']])
            df['Predicted Denial Code'] = preds

            st.success("✅ Predictions completed.")
            st.dataframe(df)

            # Download button
            st.download_button("📥 Download Results", data=df.to_csv(index=False), file_name="predicted_output.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")


import streamlit as st
import pandas as pd
import pickle

st.title("📂 Batch Denial Code Prediction")

# Load trained model and encoders
model = pickle.load(open('rf_model.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))

# Upload file
uploaded_file = st.file_uploader("Upload your .csv or .xlsx file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Encode fields
        df['CPT Code'] = encoders['CPT'].transform(df['CPT Code'])
        df['Insurance Company'] = encoders['Ins'].transform(df['Insurance Company'])
        df['Physician Name'] = encoders['Phys'].transform(df['Physician Name'])

        # Predict
        X = df[['CPT Code', 'Insurance Company', 'Physician Name', 'Payment Amount', 'Balance']]
        df['Predicted Denial Code'] = model.predict(X)

        # Show results
        st.success("✅ Predictions complete.")
        st.dataframe(df)

        # Download button
        st.download_button("📥 Download Results", df.to_csv(index=False), file_name="predicted_output.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")


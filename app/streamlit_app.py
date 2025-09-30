# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")

st.title("💳 Credit Card Fraud Detector (Demo)")
models_dir = Path("models")
available = sorted([p for p in models_dir.glob("*_pipeline.joblib")])

if not available:
    st.warning("No trained models found in ./models. Train first: "
               "`python -m src.train_model --model all --artifacts_dir .`")
else:
    model_path = st.selectbox("Select a trained model:", available, index=0)
    model = joblib.load(model_path)

    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    uploaded = st.file_uploader("Upload CSV with columns: Time, V1..V28, Amount", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        proba = model.predict_proba(df)[:, 1]
        pred = (proba >= threshold).astype(int)
        out = df.copy()
        out["fraud_proba"] = proba
        out["fraud_pred"] = pred

        st.subheader("Preview")
        st.dataframe(out.head(20))
        st.metric("Predicted Frauds (count)", int(out["fraud_pred"].sum()))
        st.metric("Mean Fraud Probability", float(np.mean(proba)))

        st.download_button("⬇️ Download predictions",
                           data=out.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv",
                           mime="text/csv")

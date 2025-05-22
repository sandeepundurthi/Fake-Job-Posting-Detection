
import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load('models/job_fraud_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit UI
st.set_page_config(page_title="Fake Job Detector", layout="wide")
st.title("🕵️ Fake Job Posting Detection")

st.markdown("""
This app uses a machine learning model to detect whether a job posting is **real** or **fake**.
Paste any job description below and click **Predict**.
""")

user_input = st.text_area("📄 Paste the job description here:", height=250)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a job description.")
    elif len(user_input.split()) < 5:
        st.warning("⚠️ Please enter a more complete job description (at least 5 words).")
    else:
        clean = clean_text(user_input)
        vec = vectorizer.transform([clean])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        confidence = max(proba)

        st.write(f"🔢 Real Probability: {proba[0]:.2%}")
        st.write(f"🔢 Fake Probability: {proba[1]:.2%}")

        if confidence < 0.6:
            st.warning("🤔 Unable to determine confidently. Please provide a more detailed description.")
        elif prediction == 1:
            st.error(f"⚠️ This job looks FAKE! (Confidence: {confidence:.2%})")
        else:
            st.success(f"✅ This job looks REAL. (Confidence: {confidence:.2%})")

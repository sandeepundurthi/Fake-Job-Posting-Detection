# 🕵️ Fake Job Posting Detection (NLP Project)

This project detects fraudulent job postings using Natural Language Processing and machine learning.

## 🚀 Features
- Trained Logistic Regression model with TF-IDF features
- Word cloud visualizations
- Streamlit Web App for live prediction
- Confidence scores for both real and fake classes

## 📁 Folder Structure
```
fake_job_detection_project/
├── app/
│   └── streamlit_app_v2.py
├── models/
│   ├── job_fraud_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── fake_job_detection_notebook.ipynb
├── README.md
├── requirements.txt
```

## ⚙️ Setup & Run
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the Streamlit App:
```bash
streamlit run app/streamlit_app_v2.py
```

## 📊 Dataset
[Kaggle: Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

## 🧠 Model
Logistic Regression trained on TF-IDF vectors with class balancing (SMOTE-ready).


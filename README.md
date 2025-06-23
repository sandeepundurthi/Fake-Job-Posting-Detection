# 🕵️ Fake Job Posting Detection (NLP Project)

This project uses Natural Language Processing (NLP) and Machine Learning to detect fraudulent job postings. It aims to assist job seekers and platforms in identifying potentially fake listings.

---

## 🚀 Features

- Binary classification: **Real vs Fake** job postings
- Preprocessing: Text cleaning, TF-IDF vectorization
- Model: **Logistic Regression** with balanced classes
- Streamlit web app for live predictions
- Word cloud visualization for both classes
- Confidence scores for prediction outputs

---

## 📁 Project Structure

```
Fake-Job-Posting-Detection/
├── app/
│   └── streamlit_app_v2.py         # Streamlit UI
├── data/
│   └── fake_job_postings.csv       # Input dataset (if included)
├── models/
│   ├── job_fraud_model.pkl         # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl        # Vectorizer for text
├── notebook/
│   └── fake_job_detection_notebook.ipynb
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- Source: [Kaggle – Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- Dataset includes fields such as title, location, description, requirements, and fraudulent label.

---

## 🧠 Model & Approach

- Model: **Logistic Regression**
- Feature Engineering:
  - Text Preprocessing
  - TF-IDF vectorization
- Class Imbalance Handling: **SMOTE**-ready
- Evaluation: Accuracy, Precision, Recall, F1-Score

---
## 🛠️ Getting Started
### 1. Clone the repository
```
# Clone the repository
git clone https://github.com/sandeepundurthi/Fake-Job-Posting-Detection.git
cd Fake-Job-Posting-Detection

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/streamlit_app_v2.py
```
---

## 👤 Author

Developed by [Sandeep Undurthi](https://github.com/sandeepundurthi)

---

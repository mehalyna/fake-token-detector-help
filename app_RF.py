import streamlit as st
import pandas as pd
import joblib
import numpy as np
from nltk import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Lowercase, remove hyphens, remove stopwords, apply Porter stemming."""
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = ENGLISH_STOP_WORDS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed = []
        for text in X:
            text = text.lower().replace('-', ' ')
            tokens = [t for t in text.split() if t.isalpha() and t not in self.stop_words]
            stems = [self.stemmer.stem(token) for token in tokens]
            processed.append(" ".join(stems))
        return np.array(processed)

@st.cache_resource
def load_models():
    rf_pipeline      = joblib.load('scam_token_detector.pkl')

    return rf_pipeline

# Load models
rf_pipeline = load_models()

st.title("Crypto Scam Token Detection")

# Sidebar model selector
model_choice = st.sidebar.selectbox(
    "Choose model",
    ["Random Forest"]
)
show_metrics = st.sidebar.checkbox("Show Test Metrics")

# User inputs
token_name = st.text_input("Token Name", "Bitcoin")
token_url  = st.text_input("Token URL", "https://crypto.com/price/bitcoin")
slug = token_url.rstrip("/").split("/")[-1]
text_input = f"{token_name} {slug}"

st.write("### Input Preview")
st.write(f"- **Name:** {token_name}")
st.write(f"- **URL Slug:** {slug}")

# Predict button
if st.button("Predict"):
    pipe = {
            "Random Forest": rf_pipeline,
            }[model_choice]
    pred = pipe.predict([text_input])[0]
    label = "SCAM" if pred >= 0.5 else "LEGIT"
    st.success(f"{model_choice} Prediction: **{label}**")
    if hasattr(pipe, "predict_proba"):
        prob = pipe.predict_proba([text_input])[0][1]
        st.write(f"Probability of scam: **{prob:.2f}**")

if show_metrics:
    st.write("## Test Set Evaluation")
    df = pd.read_csv('data/Crypto_enhanced_dataset.csv')

    slug_series = df['url'].fillna('').apply(lambda u: u.rstrip('/').split('/')[-1])
    texts = (df['name'].fillna('') + ' ' + slug_series).astype(str)
    y_true = df['is_scam']

    for label, pipe in [
        ("Random Forest", rf_pipeline)
        ]:
        st.write(f"### {label}")
        y_pred = pipe.predict(texts)
        st.text(classification_report(y_true, y_pred, digits=4))
        cm = confusion_matrix(y_true, y_pred)
        st.write("Confusion Matrix:")
        st.write(pd.DataFrame(
            cm, index=["True Legit", "True Scam"], columns=["Pred Legit", "Pred Scam"]
        ))

        fig_cm, ax_cm = plt.subplots()
        ax_cm.imshow(cm, interpolation='nearest')
        ax_cm.set_title("Confusion Matrix")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Pred Legit", "Pred Scam"])
        ax_cm.set_yticklabels(["True Legit", "True Scam"])

        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i, cm[i, j], ha="center", va="center")
        st.pyplot(fig_cm)

        probs = rf_pipeline.predict_proba(texts)[:, 1]
        fig_p, ax_p = plt.subplots()
        ax_p.hist(probs, bins=20)
        ax_p.set_title("Scam Probability Distribution")
        ax_p.set_xlabel("Predicted P(scam)")
        ax_p.set_ylabel("Count")
        st.pyplot(fig_p)




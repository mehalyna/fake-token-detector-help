import streamlit as st
import pandas as pd
import joblib
import numpy as np
from nltk import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import classification_report, confusion_matrix

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
    ext_rf_pipeline  = joblib.load('extended_nlp_rf_detector.pkl')
    lgbm_pipeline    = joblib.load('lightgbm_scam_detector.pkl')
    rf_slim_only     = joblib.load('rf_slim_only_model.pkl')
    iso_vectorizer   = joblib.load('tfidf_vectorizer.pkl')
    iso_model        = joblib.load('isolation_forest_detector.pkl')
    return rf_pipeline, ext_rf_pipeline, lgbm_pipeline, rf_slim_only, iso_vectorizer, iso_model

# Load models
rf_pipeline, ext_rf_pipeline, lgbm_pipeline, rf_slim_only, iso_vectorizer, iso_model = load_models()

st.title("Crypto Scam Token Detection")

# Sidebar model selector
model_choice = st.sidebar.selectbox(
    "Choose model",
    ["Random Forest", "Extended NLP RF", "LightGBM", "Slim RF", "IsolationForest"]
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
    if model_choice == "IsolationForest":
        vec = iso_vectorizer.transform([text_input])
        pred = iso_model.predict(vec)[0]
        label = "SCAM" if pred == -1 else "LEGIT"
        st.success(f"IsolationForest Prediction: **{label}**")
    elif model_choice == "Slim RF":
        # compute slim features on the fly
        name_length = len(token_name)
        digits_ratio = sum(c.isdigit() for c in token_name) / len(token_name) if len(token_name)>0 else 0
        hyphens_count = slug.count('-')
        X_slim = np.array([[name_length, digits_ratio, hyphens_count]])
        pred = rf_slim_only.predict(X_slim)[0]
        prob = rf_slim_only.predict_proba(X_slim)[0][1]
        label = "SCAM" if pred == 1 else "LEGIT"
        st.success(f"Slim RF Prediction: **{label}**")
        st.write(f"Probability of scam: **{prob:.2f}**")
    else:
        pipe = {
            "Random Forest": rf_pipeline,
            "Extended NLP RF": ext_rf_pipeline,
            "LightGBM": lgbm_pipeline
        }[model_choice]
        pred = pipe.predict([text_input])[0]
        label = "SCAM" if pred == 1 else "LEGIT"
        st.success(f"{model_choice} Prediction: **{label}**")
        if hasattr(pipe, "predict_proba"):
            prob = pipe.predict_proba([text_input])[0][1]
            st.write(f"Probability of scam: **{prob:.2f}**")

if show_metrics:
    st.write("## Test Set Evaluation")
    df = pd.read_csv('data/Crypto_enhanced_dataset.csv')
    # Підготуємо загальні тексти і ціль
    slug_series = df['url'].fillna('').apply(lambda u: u.rstrip('/').split('/')[-1])
    texts = (df['name'].fillna('') + ' ' + slug_series).astype(str)
    y_true = df['is_scam']

    # 1) Оцінимо текстові моделі
    for label, pipe in [
        ("Random Forest", rf_pipeline),
        ("Extended NLP RF", ext_rf_pipeline),
        ("LightGBM", lgbm_pipeline)
    ]:
        st.write(f"### {label}")
        y_pred = pipe.predict(texts)
        st.text(classification_report(y_true, y_pred, digits=4))
        cm = confusion_matrix(y_true, y_pred)
        st.write("Confusion Matrix:")
        st.write(pd.DataFrame(
            cm, index=["True Legit","True Scam"], columns=["Pred Legit","Pred Scam"]
        ))

    # 2) Оцінимо Slim-Only RF на slim-ознаках
    st.write("### Slim-Only RF")
    # Генеруємо slim-ознаки прямо в тестовому наборі
    df['name_length']   = df['name'].fillna('').str.len()
    df['digits_ratio']  = df['name'].fillna('').apply(
        lambda s: sum(c.isdigit() for c in s)/len(s) if len(s)>0 else 0
    )
    df['url_slug']      = slug_series
    df['hyphens_count'] = df['url_slug'].str.count('-')
    X_slim_test = df[['name_length','digits_ratio','hyphens_count']].values
    y_pred_slim = rf_slim_only.predict(X_slim_test)
    st.text(classification_report(y_true, y_pred_slim, digits=4))
    cm_slim = confusion_matrix(y_true, y_pred_slim)
    st.write("Confusion Matrix:")
    st.write(pd.DataFrame(
        cm_slim, index=["True Legit","True Scam"], columns=["Pred Legit","Pred Scam"]
    ))

    # 3) Оцінимо IsolationForest
    st.write("### IsolationForest")
    X_eval = iso_vectorizer.transform(texts)
    preds = iso_model.predict(X_eval)
    y_pred_iso = np.array([1 if p==-1 else 0 for p in preds])
    st.text(classification_report(y_true, y_pred_iso, digits=4))
    cm_iso = confusion_matrix(y_true, y_pred_iso)
    st.write("Confusion Matrix:")
    st.write(pd.DataFrame(
        cm_iso, index=["True Legit","True Scam"], columns=["Pred Legit","Pred Scam"]
    ))


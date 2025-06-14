import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from nltk.stem import PorterStemmer

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

def train_extended_nlp_rf(
    data_csv_path: str,
    model_output_path: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    # Load labeled data
    df = pd.read_csv(data_csv_path)
    df['url_slug'] = df['url'].fillna('').apply(lambda u: u.rstrip('/').split('/')[-1])
    df['text'] = (df['name'].fillna('') + ' ' + df['url_slug']).astype(str)
    X = df['text']
    y = df['is_scam']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Pipeline: preprocessing + feature union + RF
    pipeline = Pipeline([
        ('preproc', TextPreprocessor()),
        ('features', FeatureUnion([
            ('tfidf_trigram', TfidfVectorizer(ngram_range=(1,3), min_df=2, max_features=5000)),
            ('hash_vect', HashingVectorizer(ngram_range=(1,3), n_features=2**12, alternate_sign=False))
        ])),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # Save
    joblib.dump(pipeline, model_output_path)
    print(f"Model saved to: {model_output_path}")

if __name__ == "__main__":
    train_extended_nlp_rf(
        data_csv_path='data/Crypto_final_labeled.csv',
        model_output_path='extended_nlp_rf_detector.pkl'
    )

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def train_and_save_anomaly_detector(
        valid_csv_path: str,
        scam_csv_path: str,
        vectorizer_path: str,
        model_path: str,
        contamination: float = 0.1,
        random_state: int = 42
) -> None:
    """
    Train an IsolationForest anomaly detector on valid tokens,
    evaluate on a combined valid+scam set, and save the vectorizer and model.

    Args:
        valid_csv_path (str): Path to CSV with valid tokens (Crypto_cleaned.csv).
        scam_csv_path (str): Path to CSV with scam tokens (urls_cleaned.csv).
        vectorizer_path (str): File path to save the trained TF-IDF vectorizer.
        model_path (str): File path to save the trained anomaly detector.
        contamination (float): Proportion of anomalies expected by the model.
        random_state (int): Seed for reproducibility.
    """
    # Load data
    df_valid = pd.read_csv(valid_csv_path)
    df_scam = pd.read_csv(scam_csv_path)

    # Rename 'crypturl' to 'url' for consistency
    df_valid = df_valid.rename(columns={'crypturl': 'url'})

    # Prepare combined text feature
    for df in (df_valid, df_scam):
        df['url_slug'] = df['url'].fillna('').apply(lambda u: u.rstrip('/').split('/')[-1])
        df['text_feature'] = (df['name'].fillna('') + ' ' + df['url_slug']).astype(str)

    # Fit TF-IDF on valid tokens
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_valid = vectorizer.fit_transform(df_valid['text_feature'])

    # Train IsolationForest
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(X_valid)

    # Prepare evaluation set
    df_eval = pd.concat([df_valid, df_scam], ignore_index=True)
    X_eval = vectorizer.transform(df_eval['text_feature'])
    y_true = [0] * len(df_valid) + [1] * len(df_scam)

    # Predict: -1 = anomaly (scam), 1 = normal (valid)
    preds = iso.predict(X_eval)
    y_pred = [1 if p == -1 else 0 for p in preds]

    # Output metrics
    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, digits=4))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))

    # Save vectorizer and model
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(iso, model_path)
    print(f"TF-IDF vectorizer saved to: {vectorizer_path}")
    print(f"Anomaly detector saved to: {model_path}")


if __name__ == "__main__":
    train_and_save_anomaly_detector(
        valid_csv_path='data/Crypto_cleaned.csv',
        scam_csv_path='data/urls_cleaned.csv',
        vectorizer_path='tfidf_vectorizer.pkl',
        model_path='isolation_forest_detector.pkl',
        contamination=0.1,
        random_state=42
    )

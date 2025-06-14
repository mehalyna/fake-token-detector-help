import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
import joblib

def train_and_evaluate_lightgbm(
    data_csv_path: str,
    model_output_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    """
    Train a LightGBM classifier on scam token data and save the model pipeline.

    Args:
        data_csv_path (str): Path to CSV containing 'name', 'url', and 'is_scam'.
        model_output_path (str): Path to save the trained pipeline.
        test_size (float): Fraction of data to use for the test set.
        random_state (int): Seed for reproducibility.
    """
    # 1. Load data
    df = pd.read_csv(data_csv_path)

    # 2. Build text feature: name + URL slug
    df['url_slug'] = df['url'].fillna('').apply(lambda u: u.rstrip('/').split('/')[-1])
    df['text_feature'] = (df['name'].fillna('') + ' ' + df['url_slug']).astype(str)

    X = df['text_feature']
    y = df['is_scam']

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 4. Pipeline: TF-IDF + LightGBM
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2)),
        ('lgbm', LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    # 5. Train
    pipeline.fit(X_train, y_train)

    # 6. Predict & evaluate
    y_pred = pipeline.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # 7. Save model
    joblib.dump(pipeline, model_output_path)
    print(f"Saved LightGBM pipeline to: {model_output_path}")

if __name__ == "__main__":
    DATA_PATH = "data/Crypto_final_labeled.csv"
    MODEL_PATH = "lightgbm_scam_detector.pkl"
    train_and_evaluate_lightgbm(
        data_csv_path=DATA_PATH,
        model_output_path=MODEL_PATH,
        test_size=0.2,
        random_state=42
    )

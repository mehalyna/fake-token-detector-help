import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

def train_and_save_scam_detector(
    data_csv_path: str,
    model_output_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    """
    Train a scam token detection model and save the pipeline to disk.

    Args:
        data_csv_path (str): Path to the labeled CSV with 'name', 'url', and 'is_scam'.
        model_output_path (str): File path to save the trained model pipeline.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.
    """
    # 1. Load the labeled dataset
    df = pd.read_csv(data_csv_path)

    # 2. Prepare the text feature by combining name and URL slug
    df['url_slug'] = df['url'].fillna('').apply(lambda u: u.rstrip('/').split('/')[-1])
    df['text_feature'] = (df['name'].fillna('') + ' ' + df['url_slug']).astype(str)

    # 3. Define features and target
    X = df['text_feature']
    y = df['is_scam']

    # 4. Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 5. Create a pipeline with TF-IDF vectorizer and RandomForest classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1))
    ])

    # 6. Train the model
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 7. Save the trained pipeline
    joblib.dump(pipeline, model_output_path)

    print(f"Model trained and saved to: {model_output_path}")


if __name__ == "__main__":
    # Example call when running this script directly in PyCharm
    DATA_PATH = "data/Crypto_final_labeled.csv"
    MODEL_PATH = "scam_token_detector.pkl"
    train_and_save_scam_detector(
        data_csv_path=DATA_PATH,
        model_output_path=MODEL_PATH,
        test_size=0.2,
        random_state=42
    )

# Example usage in Streamlit:
# train_and_save_scam_detector(
#     data_csv_path='/mnt/data/Crypto_final_labeled_unified_url.csv',
#     model_output_path='scam_token_detector.pkl'
# )

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def train_and_save_rf_slim(
    data_csv_path: str,
    model_output_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    """
    Train a RandomForest model on the enhanced dataset with slim and numeric features, and save it.
    """
    # Load data
    df = pd.read_csv(data_csv_path)

    # Prepare text feature
    df['url_slug'] = df['url'].fillna('').apply(lambda u: u.rstrip('/').split('/')[-1])
    df['text_feature'] = (df['name'].fillna('') + ' ' + df['url_slug']).astype(str)

    # Define feature columns
    numeric_features = ['price', 'volume24hrs', 'marketcap', 'circulatingsupply', 'maxsupply', 'totalsupply']
    slim_features    = ['name_length', 'digits_ratio', 'hyphens_count']

    X = df[numeric_features + slim_features + ['text_feature']]
    y = df['is_scam']

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Build ColumnTransformer for numeric and text
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features + slim_features),
        ('text', TfidfVectorizer(ngram_range=(1,2), min_df=2), 'text_feature')
    ])

    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=random_state
        ))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print('=== Classification Report ===')
    print(classification_report(y_test, y_pred, digits=4))
    print('=== Confusion Matrix ===')
    print(confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(pipeline, model_output_path)
    print(f'Model saved to {model_output_path}')


if __name__ == '__main__':
    train_and_save_rf_slim(
        data_csv_path='data/Crypto_enhanced_dataset.csv',
        model_output_path='rf_slim_model.pkl'
    )
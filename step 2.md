### 4. Навчання ML-моделі

| Категорія            | Деталі                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Що підготувати**   | 1. Файл `data/processed/tokens_features.csv` із фічами та мітками.<br>2. Новий скрипт `train_model.py` у корені проєкту.<br>3. README з короткою інструкцією: як запускати скрипт і де шукати збережену модель („model.pkl“).                                                                                                                                                                                                                           |
| **Що говорити**      | 1. “Тепер ми навчимо дві базові моделі — Random Forest і Logistic Regression — і порівняємо їхню продуктивність на нашому наборі фіч.”<br>2. “Розіб’ємо дані на train/test (80/20), щоб оцінити узагальнювальну здатність.”<br>3. “Підбирати гіперпараметри зараз не будемо — візьмемо дефолтні, аби показати загальний процес.”<br>4. “Наприкінці збережемо найкращу модель у файл і подивимося на базові метрики: accuracy, precision, recall та F1.” |
| **Що демонструвати** | **Файл `train_model.py`:**                                                                                                                                                                                                                                                                                                                                                                                                                              |

```python
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble       import RandomForestClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- 1. Load features dataset ---
df = pd.read_csv("data/processed/tokens_features.csv")

# Split into X and y
X = df.drop(columns=["label", "name", "crypturl", "any_url", "domain", "date_taken"])
y = df["label"]

# --- 2. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. Define models ---
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1)
}

results = {}

# --- 4. Train and evaluate ---
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
    }
    print(f"--- {name} ---")
    print("Accuracy: ", results[name]["accuracy"])
    print("Precision:", results[name]["precision"])
    print("Recall:   ", results[name]["recall"])
    print("F1-score: ", results[name]["f1"])
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

# --- 5. Save best model ---
best_model_name = max(results, key=lambda k: results[k]["f1"])
best_model = models[best_model_name]
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"✅ Best model ({best_model_name}) saved to model.pkl")
```

1. Поясніть, як вибрали колонки для X (відкинули ідентифікатори та небінарні рядкові поля).
2. Запустіть у терміналі:

   ```bash
   py -3 train_model_RF.py
   ```
3. Продемонструйте вивід з метриками для обох моделей та матрицю невідповідностей.
4. Поясніть учасникам, що в `model.pkl` зберігається об’єкт моделі для подальшого використання в Streamlit. |

---

Після цього всі матимуть попередньо навчену модель. Далі переходимо до етапу 5 — візуалізації (barplot важливості ознак та матриця невідповідностей) і інтеграції в Streamlit.

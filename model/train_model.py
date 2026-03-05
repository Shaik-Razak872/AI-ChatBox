import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from preprocess import clean_text   # preprocess.py is in the same folder

# ===============================
# 1. Load Dataset
# ===============================
data = pd.read_csv("dataset/sentiment_data.csv")

# ===============================
# 2. Preprocess Text
# ===============================
data['text'] = data['text'].apply(clean_text)

X = data['text']
y = data['sentiment']

# ===============================
# 3. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 4. Build ML Pipeline (UPDATED TF-IDF)
# ===============================
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3),
        stop_words='english',
        max_features=8000
    )),
    ('classifier', LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    ))
])

# ===============================
# 5. Train Model
# ===============================
model.fit(X_train, y_train)

# ===============================
# 6. Evaluate Model
# ===============================
y_pred = model.predict(X_test)

print("\n📊 Model Evaluation Results")
print("----------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 7. Save Trained Model
# ===============================
with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Advanced ML Model Trained & Saved Successfully")

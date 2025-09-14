import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
file_url = "https://drive.google.com/uc?export=download&id=1RfDP3sOk6Ce5sIP-F1ayd3WsH0hwdEf9"
df = pd.read_csv(file_url)

print(" Dataset Loaded. Shape:", df.shape)
print("\n First 5 rows:\n", df.head())

df.columns = df.columns.str.strip()

print("\n Columns in dataset:\n", df.columns.tolist())

feature_cols = [
    "Movie_genre_top1",
    "Series_genre_top1",
    "Binge frequency per week",
    "Screen Time Movies/series in hours per week  \n(Provide value between 0-40)" 
]
target_col = "Ott Top1"

# Check if all columns exist
for col in feature_cols + [target_col]:
    if col not in df.columns:
        raise KeyError(f"Column missing: {col}")

X = df[feature_cols].copy()
y = df[target_col].copy()

X["Binge frequency per week"] = pd.to_numeric(
    X["Binge frequency per week"], errors="coerce"
)

# Clean Screen Time column (convert "7 to 8 hrs" → 8, "5 hrs" → 5)
def clean_hours(val):
    if pd.isna(val):
        return np.nan
    val = str(val).lower().strip()
    if "to" in val:  # e.g., "7 to 8 hrs"
        parts = [p for p in val.replace("hrs", "").split("to") if p.strip().isdigit()]
        if len(parts) == 2:
            return float(parts[1])  # take upper bound
    if "hr" in val:
        val = val.replace("hrs", "").replace("hr", "").strip()
    try:
        return float(val)
    except:
        return np.nan

X["Screen Time Movies/series in hours per week  \n(Provide value between 0-40)"] = \
    X["Screen Time Movies/series in hours per week  \n(Provide value between 0-40)"].apply(clean_hours)

# Drop rows with missing values after cleaning
X = X.dropna()
y = y.loc[X.index]

print("\n Data cleaned successfully.")
print("X shape after cleaning:", X.shape)
print("y shape after cleaning:", y.shape)

label_encoders = {}

for col in ["Movie_genre_top1", "Series_genre_top1"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode Target (OTT Platform)
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y.astype(str))

print("\n Encoded Target Classes:", target_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n Training data shape:", X_train.shape)
print(" Testing data shape:", X_test.shape)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# Confusion Matrix Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
            cmap="Blues", xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

sample_idx = X_test.iloc[[0]]
predicted_label = target_encoder.inverse_transform(model.predict(sample_idx))[0]
print("\n Example Prediction:", predicted_label)

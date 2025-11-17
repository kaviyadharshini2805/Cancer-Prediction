import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load dataset
df = pd.read_csv("cancer_data.csv")

# Features & target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)  # smaller K for small dataset
knn.fit(X_train_scaled, y_train)

# Save model, scaler, and feature names
joblib.dump(knn, "knn_cancer_model.pkl")
joblib.dump(scaler, "knn_scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("✅ Model saved")
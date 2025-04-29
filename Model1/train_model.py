import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.metrics import classification_report, confusion_matrix # type: ignore
import joblib # type: ignore

# Load the dataset
df = pd.read_csv("context_aware_speed_fine_dataset.csv")

# Define features and target
X = df.drop(columns=["fine_probability"])
y = (df["fine_probability"] > 0.7).astype(int)  # 1 = Fine Likely, 0 = Fine Unlikely

# Categorical and numerical columns
categorical_cols = ["time_of_day", "weather", "road_type", "driver_history"]
numeric_cols = ["speed_limit", "vehicle_speed", "traffic_density", "area_sensitivity"]

# Preprocessing for categorical features
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(drop="first"), categorical_cols)
], remainder="passthrough")

# Create a pipeline with preprocessing and logistic regression
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))



joblib.dump(model, 'model.pkl')

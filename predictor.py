import joblib
import pandas as pd
import numpy as np
import os

best_model = joblib.load("best_model.pkl")
all_models = joblib.load("all_models.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("encoders.pkl")
X_scaled, y = joblib.load("training_data.pkl")
performance_df = pd.read_csv("model_performance.csv")

ensemble_model = None
if os.path.exists("ensemble_model.pkl"):
    ensemble_model = joblib.load("ensemble_model.pkl")

data = pd.read_csv("healthcare_synthetic_data.csv")
TARGET_COLUMN = "Heart_Disease_Risk"

for col in data.columns:
    if "id" in col.lower() or "pid" in col.lower():
        data = data.drop(col, axis=1)

feature_names = data.drop(TARGET_COLUMN, axis=1).columns


def get_model(choice):
    if choice == "Best Model":
        return best_model
    elif choice == "Ensemble Model" and ensemble_model:
        return ensemble_model
    else:
        return all_models.get(choice, best_model)


def preprocess(input_dict):
    df = pd.DataFrame([input_dict])

    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    df = df[feature_names]
    return scaler.transform(df)


def predict_risk(input_dict, model_choice):
    model = get_model(model_choice)
    scaled = preprocess(input_dict)
    return model.predict_proba(scaled)[0][1]


def get_confusion_matrix(choice):
    from sklearn.metrics import confusion_matrix
    model = get_model(choice)
    preds = model.predict(X_scaled)
    return confusion_matrix(y, preds)

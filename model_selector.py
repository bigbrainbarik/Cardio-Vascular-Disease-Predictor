import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ===============================
# LOAD DATA
# ===============================

data = pd.read_csv("healthcare_synthetic_data.csv")
TARGET_COLUMN = "Heart_Disease_Risk"

for col in data.columns:
    if "id" in col.lower() or "pid" in col.lower():
        data = data.drop(col, axis=1)

X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN]

label_encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

feature_names = X.columns

# ===============================
# MODELS
# ===============================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier(max_iter=800),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "LightGBM": LGBMClassifier(verbose=-1),
    "CatBoost": CatBoostClassifier(verbose=0)
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

plt.figure(figsize=(10, 8))

# ===============================
# CROSS VALIDATION
# ===============================

for name, model in models.items():
    print(f"Training {name}")
    start = time.time()

    roc_scores = []
    f1_scores = []
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train = X_scaled.iloc[train_idx]
        X_test = X_scaled.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        roc_scores.append(roc_auc_score(y_test, y_prob))
        f1_scores.append(f1_score(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_auc = np.mean(roc_scores)
    mean_f1 = np.mean(f1_scores)

    plt.plot(mean_fpr, np.mean(tprs, axis=0),
             label=f"{name} (AUC={mean_auc:.3f})")

    results.append([
        name,
        mean_auc,
        mean_f1,
        time.time() - start
    ])

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# ===============================
# PRECISION-RECALL (Combined)
# ===============================

plt.figure(figsize=(10, 8))

for name, model in models.items():
    model.fit(X_scaled, y)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_prob)
    plt.plot(recall, precision, label=name)

plt.title("Precision-Recall Curve Comparison")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.show()

# ===============================
# RESULTS TABLE
# ===============================

results_df = pd.DataFrame(
    results,
    columns=["Model", "ROC-AUC", "F1 Score", "Training Time"]
).sort_values(by=["ROC-AUC", "F1 Score"], ascending=False)

print("\nModel Performance:")
print(results_df)

# ===============================
# TRAIN FULL MODELS
# ===============================

for name, model in models.items():
    model.fit(X_scaled, y)

joblib.dump(models, "all_models.pkl")

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
joblib.dump(best_model, "best_model.pkl")

# ===============================
# CONFUSION MATRIX HEATMAP
# ===============================

y_pred_best = best_model.predict(X_scaled)
cm = confusion_matrix(y, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Best Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================
# CALIBRATION CURVE
# ===============================

prob_true, prob_pred = calibration_curve(
    y,
    best_model.predict_proba(X_scaled)[:, 1],
    n_bins=10
)

plt.figure()
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("Calibration Curve (Best Model)")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.show()

# ===============================
# THRESHOLD OPTIMIZATION
# ===============================

y_probs = best_model.predict_proba(X_scaled)[:, 1]
thresholds = np.linspace(0, 1, 100)
f1_scores = [f1_score(y, (y_probs >= t)) for t in thresholds]

optimal_threshold = thresholds[np.argmax(f1_scores)]
print("Optimal Threshold for Best F1:", optimal_threshold)

# ===============================
# FEATURE IMPORTANCE HEATMAP
# ===============================

if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        importance_df.set_index("Feature"),
        annot=True,
        cmap="viridis"
    )
    plt.title("Feature Importance Heatmap")
    plt.show()

# ===============================
# SHAP
# ===============================

print("Generating SHAP plots...")

try:
    if hasattr(best_model, "feature_importances_"):
        explainer = shap.TreeExplainer(best_model)
    else:
        explainer = shap.Explainer(best_model, X_scaled)

    shap_values = explainer(X_scaled)

    shap.summary_plot(shap_values, X_scaled)
    shap.summary_plot(shap_values, X_scaled, plot_type="bar")

except Exception as e:
    print("SHAP failed:", e)

# ===============================
# SAVE ARTIFACTS
# ===============================

top_models = [
    name for name in results_df.head(3)["Model"].values
    if name != "CatBoost"
]

if len(top_models) >= 2:
    ensemble = VotingClassifier(
        estimators=[(n, models[n]) for n in top_models],
        voting="soft"
    )
    ensemble.fit(X_scaled, y)
    joblib.dump(ensemble, "ensemble_model.pkl")

joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "encoders.pkl")
joblib.dump((X_scaled, y), "training_data.pkl")
results_df.to_csv("model_performance.csv", index=False)

print("All analysis complete and saved.")

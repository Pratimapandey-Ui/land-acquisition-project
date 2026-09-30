"""
5_train_final_model.py
------------------------
Run: python src/5_train_final_model.py
Trains XGBoost -- our FINAL model, compared against the baseline.
(Note: this uses real XGBoost -- make sure `pip install xgboost` worked
on your machine. In the sandbox we used a scikit-learn stand-in because
that environment had no internet access; on your own machine this will
run the real thing.)
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os
 
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
 
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
 
model = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, eval_metric="logloss",
    random_state=42
)
 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
print("5-fold CV F1 scores:", np.round(cv_scores, 3))
print("CV F1 mean/std:", round(cv_scores.mean(), 3), "/", round(cv_scores.std(), 3))
 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
 
print("\n=== FINAL MODEL: XGBoost ===")
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall   :", round(recall_score(y_test, y_pred), 3))
print("F1       :", round(f1_score(y_test, y_pred), 3))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_proba), 3))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
 
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgboost_model.pkl")
print("\nSaved: models/xgboost_model.pkl")
 
# quick feature importance printout
importances = pd.Series(model.feature_importances_, index=X_train.columns)
print("\nTop 10 feature importances:")
print(importances.sort_values(ascending=False).head(10))
 
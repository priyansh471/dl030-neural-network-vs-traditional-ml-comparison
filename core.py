"""Model code for the DL-030 demo app (no Streamlit imports, so it can be tested alone).

Same data, same stratified 80/20 split (seed 42) and same scaling rule as the notebook:
the scaler is fitted on the training set only.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
CLASS_NAMES = {0: "malignant", 1: "benign"}
MLP_NAME = "Neural Network (MLP)"


def load_data():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, stratify=data.target, random_state=SEED
    )
    return data, X_train, X_test, y_train, y_test


def build_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)
        ),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
        # scikit-learn version of the notebook's Keras MLP: 30 -> 32 -> 16 -> 1, ReLU, Adam 1e-3,
        # batch 16, early stopping on a 20% validation split with patience 15. No dropout layer.
        MLP_NAME: make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                learning_rate_init=1e-3,
                batch_size=16,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=15,
                random_state=SEED,
            ),
        ),
    }


def train_all():
    """Fit every model on the training split. Returns (models, train_times, data tuple)."""
    data, X_train, X_test, y_train, y_test = load_data()
    models, times = build_models(), {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        times[name] = time.time() - t0
    return models, times, (data, X_train, X_test, y_train, y_test)


def evaluate(models, times, X_test, y_test):
    rows = []
    for name, model in models.items():
        pred = model.predict(X_test)
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred),
                "Recall": recall_score(y_test, pred),
                "F1": f1_score(y_test, pred),
                "Missed malignant": int(((y_test == 0) & (pred == 1)).sum()),
                "False alarms": int(((y_test == 1) & (pred == 0)).sum()),
                "Train time (s)": times[name],
            }
        )
    return pd.DataFrame(rows)


def predict_all(models, x):
    """Predict one sample (1-D array of 30 values) with every model."""
    x = np.asarray(x, dtype=float).reshape(1, -1)
    rows = []
    for name, model in models.items():
        proba_benign = float(model.predict_proba(x)[0, 1])
        label = int(model.predict(x)[0])
        rows.append(
            {
                "Model": name,
                "Prediction": CLASS_NAMES[label],
                "P(benign)": proba_benign,
                "Confidence": max(proba_benign, 1 - proba_benign),
            }
        )
    return pd.DataFrame(rows)

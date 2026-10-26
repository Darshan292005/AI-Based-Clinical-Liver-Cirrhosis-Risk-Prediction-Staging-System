import os
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import joblib
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE


# User specified features
NUMERICAL_FEATURES = [
    "Age",
    "Bilirubin",
    "Albumin",
    "Platelets",
    "SGOT",
    "Prothrombin Time",
]
CATEGORICAL_FEATURES = ["Sex", "Ascites", "Hepatomegaly", "Edema"]
TARGET = "Stage"


def load_and_preprocess_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)

    # Check column names. OpenML might use lowercase or slightly different names.
    col_mapping = {
        "age": "Age",
        "sex": "Sex",
        "ascites": "Ascites",
        "hepatomegaly": "Hepatomegaly",
        "hepato": "Hepatomegaly",
        "edema": "Edema",
        "bilirubin": "Bilirubin",
        "bili": "Bilirubin",
        "albumin": "Albumin",
        "platelets": "Platelets",
        "platelet": "Platelets",
        "prothrombin": "Prothrombin Time",
        "protime": "Prothrombin Time",
        "sgot": "SGOT",
        "ast": "SGOT",
        "stage": "Stage",
        "class": "Stage",
    }
    df.rename(columns=lambda x: col_mapping.get(x.lower(), x.title()), inplace=True)

    # Print the available columns
    print("Available columns:", df.columns.tolist())

    # Convert 'Age' from days to years if necessary
    if "Age" in df.columns and df["Age"].max() > 200:
        df["Age"] = df["Age"] / 365.25

    # Filter columns to only what is required
    available_features = [
        f for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if f in df.columns
    ]

    if TARGET not in df.columns:
        # Sometimes class is 'Class' or 'Status', we need to check precisely.
        print("Error: Target column not found.")
        return None, None

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET])

    # OpenML PBC class is sometimes '1', '2', '3', '4'. Let's ensure it's integer
    df[TARGET] = df[TARGET].astype(int)

    X = df[available_features]
    y = df[TARGET]

    return X, y


def build_pipeline(numeric_features, categorical_features, model):
    # Preprocessing for numerical data
    numeric_transformer = ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing for categorical data
    categorical_transformer = ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Bundle preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Bundle preprocessor & model
    clf = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ]
    )

    return clf


def evaluate_model(y_test, y_pred, y_pred_proba, model_name):
    print(f"\n--- {model_name} Evaluation ---")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}


def train_and_evaluate():
    data_path = "data/raw_cirrhosis.csv"
    if not os.path.exists(data_path):
        if os.path.exists("cirrhosis.csv"):
            data_path = "cirrhosis.csv"
        else:
            print("Data file not found. Please run download_data.py first.")
            return


    X, y = load_and_preprocess_data(data_path)
    if X is None:
        return

    num_cols = [col for col in NUMERICAL_FEATURES if col in X.columns]
    cat_cols = [col for col in CATEGORICAL_FEATURES if col in X.columns]

    # For XGBoost: target labels must be 0-indexed
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Save the label encoder classes to map back later
    os.makedirs("models", exist_ok=True)
    joblib.dump(label_encoder, "models/label_encoder.joblib")

    # Train, validation, test split: 70%, 15%, 15%
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1764,
        random_state=42,
        stratify=y_train_val,
    )  # 0.1764 of 0.85 approx 0.15

    print(
        f"Train set: {X_train.shape[0]}, Validation set: {X_val.shape[0]}, Test set: {X_test.shape[0]}"
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
        "SVM": SVC(probability=True, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42),

    }

    best_model_name = ""
    best_acc = 0
    best_pipeline = None

    results = {}

    for name, model in list(models.items()):
        pipeline = build_pipeline(num_cols, cat_cols, model)
        pipeline.fit(X_train, y_train)

        y_val_pred = pipeline.predict(X_val)
        y_val_proba = pipeline.predict_proba(X_val)

        eval_metrics = evaluate_model(y_val, y_val_pred, y_val_proba, name)
        results[name] = eval_metrics

        if eval_metrics["Accuracy"] > best_acc:
            best_acc = eval_metrics["Accuracy"]
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nBest Model: {best_model_name} with Validation Accuracy: {best_acc:.4f}")

    # Retrain on Train+Val and test on Test
    print("\nRetraining best model on Train+Validation combined set...")
    final_pipeline = build_pipeline(num_cols, cat_cols, models[best_model_name])
    final_pipeline.fit(X_train_val, y_train_val)

    y_test_pred = final_pipeline.predict(X_test)
    y_test_proba = final_pipeline.predict_proba(X_test)

    evaluate_model(y_test, y_test_pred, y_test_proba, f"{best_model_name} (Final Test)")

    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title(f"Confusion Matrix: {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/confusion_matrix.png")
    plt.close()

    # Feature Importance (if applicable)
    if best_model_name in ["Random Forest", "XGBoost"]:
        # Extract feature names from preprocessor
        preprocessor = final_pipeline.named_steps["preprocessor"]
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        all_features = num_cols + list(cat_feature_names)

        model = final_pipeline.named_steps["model"]
        importances = model.feature_importances_

        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(all_features)), importances[indices], align="center")
        plt.xticks(
            range(len(all_features)),
            [all_features[i] for i in indices],
            rotation=45,
            ha="right",
        )
        plt.xlim(-1, len(all_features))
        plt.tight_layout()
        plt.savefig("visualizations/feature_importance.png")
        plt.close()
        print("\nFeature importance chart saved.")

    # Save the best pipeline
    joblib.dump(final_pipeline, "models/best_model_pipeline.joblib")
    print("Model and pipeline successfully saved to models/best_model_pipeline.joblib")


if __name__ == "__main__":
    train_and_evaluate()

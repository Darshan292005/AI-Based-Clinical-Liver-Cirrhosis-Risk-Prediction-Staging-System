# AI-Based Liver Cirrhosis Prediction & Staging System

## 🌟 Overview
This project presents an end-to-end AI-powered clinical decision support system that predicts and classifies liver cirrhosis stages (Stage 1 to 4) based on patient clinical and biochemical biomarkers. 

The core predictive engine leverages **Scikit-Learn** and **Imbalanced-Learn (SMOTE)** to handle real-world clinical class imbalances, evaluates multiple competitive algorithms (**Random Forest**, **SVM**, and **XGBoost**), and persists the optimal trained pipeline for production deployment. The inference engine is integrated into a **Django** web application paired with an interactive medical chatbot and clinical recommendation assistant.

---

## 💻 Project Structure
```
cirrhosis_project/
│
├── config/                  # Django project configuration (settings.py, urls.py)
├── predictor/               # Main prediction app
│   ├── views.py             # ML inference engine & clinical suggestion rules
│   └── urls.py              # Routing for UI and /predict API
├── chatbot/                 # Medical assistant chatbot app
│   ├── views.py             # Chatbot question-answering logic
│   └── urls.py              # Routing for /chat/ask_chatbot
├── templates/predictor/     # Frontend UI templates
│   ├── index.html           # Patient biomarker submission form
│   └── result.html          # Interactive report with chatbot QA
├── data/                    # Dataset directory
│   ├── raw_cirrhosis.csv    # Mayo Clinic PBC training dataset
│   └── collected_user_data.csv # Logged patient test entries
├── models/                  # Serialized machine learning artifacts
│   ├── best_model_pipeline.joblib  # Full scikit-learn preprocessing & model pipeline
│   └── label_encoder.joblib        # Target label encoder
├── visualizations/          # Generated model diagnostic graphics
│   ├── confusion_matrix.png # Confusion matrix plot
│   └── feature_importance.png # Feature importances plot
│
├── download_data.py         # Downloads Mayo Clinic dataset from verified repository
├── train_model.py           # Preprocesses data, runs SMOTE, benchmarks models, saves artifacts
├── manage.py                # Django CLI utility
├── requirements.txt         # Project dependencies
├── .gitignore               # Version control exclusion rules
└── README.md                # Project documentation
```

---

## 🔬 Machine Learning Pipeline & Architecture Workflow

```
[Patient Data] ──> [Imputer & Scaler] ──> [SMOTE Sampling] ──> [Model Inference] ──> [Stage & Confidence]
                         │                                              │
               Median / Mode Imputation                         XGBoost / Random Forest
               + OneHot Encoding                                + Softmax Probability
```

1. **Data Preprocessing & Encoding**:
   - Numerical biomarkers (`Age`, `Bilirubin`, `Albumin`, `Platelets`, `SGOT`, `Prothrombin Time`) are imputed using **Median Imputation** and standardized via `StandardScaler`.
   - Categorical parameters (`Sex`, `Ascites`, `Hepatomegaly`, `Edema`) are imputed with **Most-Frequent Imputation** and encoded via `OneHotEncoder(handle_unknown='ignore')`.
2. **Handling Class Imbalance**:
   - Clinical datasets often have uneven distributions of disease stages. We apply **SMOTE** (Synthetic Minority Over-sampling Technique) inside an `imblearn.pipeline.Pipeline` during training to ensure the classifier does not bias towards the majority class.
3. **Multi-Model Evaluation**:
   - Compares **Random Forest Classifier**, **Support Vector Classifier (SVC)**, and **XGBoost Classifier**.
   - Evaluates Accuracy, Precision, Recall, and F1-score on isolated validation and test splits.
4. **Pipeline Serialization**:
   - The winning pipeline and its corresponding `LabelEncoder` are serialized into `models/best_model_pipeline.joblib` and `models/label_encoder.joblib`.
5. **Django Production Deployment**:
   - [predictor/views.py](file:///d:/project/project%20demo/cirrhosis_project/predictor/views.py) loads the joblib pipeline into memory.
   - Incoming patient data is structured into a Pandas DataFrame and passed directly into `model_pipeline.predict()` and `model_pipeline.predict_proba()`.
   - Generates exact class predictions (Stages 1–4), genuine model confidence percentages, and dynamic clinical suggestions.

---

## 🛠 Instructions to Run Locally

### 1. Setup Virtual Environment and Install Dependencies
Open your PowerShell or command terminal in `cirrhosis_project`:

```powershell
# Using standard Python
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Data & Train the ML Pipeline
Train and export the machine learning artifacts:
```powershell
python download_data.py
python train_model.py
```
> Running `train_model.py` compares models, saves diagnostic plots in `visualizations/`, and saves the production pipeline in `models/`.

### 3. Start the Web Application
```powershell
python manage.py runserver
```
Open your browser and navigate to: **`http://127.0.0.1:8000/`**

---

## 🩺 Sample Inputs for Testing

Input the following test profile through the web form to simulate an **Advanced Stage** patient:

* **Age:** `55`
* **Sex:** `Female (F)`
* **Bilirubin (mg/dL):** `14.5` *(Significantly elevated)*
* **Albumin (g/dL):** `2.6` *(Low - impaired liver synthesis)*
* **Platelets (cc/1000):** `95` *(Thrombocytopenia)*
* **SGOT (U/L):** `130.5` *(Active hepatic cellular damage)*
* **Prothrombin Time (s):** `14.8` *(Prolonged clotting duration)*
* **Ascites:** `Yes (Y)`
* **Hepatomegaly:** `Yes (Y)`
* **Edema:** `Present (Y)`

Click **"Start Prediction"** — the AI pipeline will classify the condition as **Stage 4 (Critical Risk)**, provide biomarker explanations, and enable targeted questions with the interactive chatbot.

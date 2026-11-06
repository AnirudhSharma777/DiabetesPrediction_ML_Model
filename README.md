# Diabetes Prediction Model

## Project overview

This is a beginner's binary-classification project built around a local
`diabetes.csv` file. The intended objective is to use eight numeric health and
demographic measurements to predict `Outcome`:

- `0`: the record is labeled as not having diabetes
- `1`: the record is labeled as having diabetes

This repository is a learning project, not a medical product. It contains an
exploration notebook, a reproducible training script that compares seven
classifiers and saves the best one, and an interactive Streamlit web app that
demonstrates how the model works.

## Project files

| File | What it does |
|---|---|
| `diabetes.csv` | Local dataset. |
| `DiabetesPredictionModel.ipynb` | Exploration notebook: EDA, outlier checks and a first model comparison. |
| `diabetes_model.py` | Shared code: feature list, preprocessing pipeline and candidate models. |
| `train.py` | Trains, compares and evaluates the models; saves `model/diabetes_model.joblib` and `model/metrics.json`. |
| `app.py` | Streamlit web app to try the model, view its performance and explore the data. |
| `model/` | Saved pipeline (preprocessing + model) and evaluation metrics used by the app. |
| `main.py` | Opens the CSV in D-Tale for interactive exploration. |
| `requirements.txt` | Python dependencies. |

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python train.py        # retrain + save the model and metrics (takes a few seconds)
streamlit run app.py   # open the demo at http://localhost:8501
```

The trained model is committed in `model/`, so `streamlit run app.py` works
without retraining. If the files are missing the app trains the model on
first start.

### Share the demo with anyone

The app runs on [Streamlit Community Cloud](https://streamlit.io/cloud) for
free: sign in with GitHub, click **Create app**, pick this repository, branch
`main` and main file `app.py`, then deploy. You get a public URL you can send
to anyone.

## The web app

- **Try the model** – enter the eight measurements (or load an example
  profile) and see the predicted probability on a gauge, the yes/no decision,
  and a chart showing how much each input raises or lowers the risk compared
  with a typical (median) value.
- **Model performance** – held-out test metrics, confusion matrix, ROC curve,
  a decision-threshold slider showing the precision/recall trade-off, the
  cross-validated comparison of all models and permutation feature importance.
- **Explore the data** – class balance and per-feature distributions for
  diabetic vs. non-diabetic records.
- **How it works** – the training pipeline in plain language.

## Dataset

The CSV currently has **769 rows and 9 columns**, including the target.
Its column names are consistent with a commonly used Pima-style diabetes
dataset, but this repository does not document the dataset's source, license,
collection process, or intended population. Record and verify those details
before sharing or using the data beyond this learning exercise.

| Column | Meaning in the dataset |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Glucose measurement |
| `BloodPressure` | Blood pressure measurement |
| `SkinThickness` | Skin thickness measurement |
| `Insulin` | Insulin measurement |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Dataset-specific diabetes pedigree score |
| `Age` | Age |
| `Outcome` | Binary target label (`0` or `1`) |

The CSV has no explicit null values, but some measurement columns contain
zeroes that may represent missing or invalid measurements. Before duplicate
removal, the observed class counts are 500 records with `Outcome = 0` and 269
with `Outcome = 1` (about 65% / 35%). The main notebook drops one duplicate,
leaving 768 rows. These class counts describe this file, not the prevalence of
diabetes in a population.

## Training pipeline (`train.py`)

1. Load `diabetes.csv` and drop exact duplicate rows (768 rows remain).
2. Stratified 80/20 train/test split (`random_state=42`). The test set is used
   only once, at the end.
3. Preprocessing inside a scikit-learn `Pipeline`, fitted on training data
   only (no test-set leakage):
   - zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`
     are treated as missing and replaced with the training-set median;
   - standard scaling for models that need it (Logistic Regression, SVM, KNN).
4. Compare Logistic Regression, Decision Tree, Random Forest, Gradient
   Boosting, SVM, KNN and XGBoost with 5-fold stratified cross-validation on
   the training set, ranked by ROC-AUC.
5. Refit the best model on the full training set, evaluate once on the test
   set and compute permutation feature importance.
6. Save the whole pipeline with `joblib` plus all metrics as JSON.

### Current results

Cross-validated ROC-AUC on the training set (5 folds):

| Model | ROC-AUC |
|---|---|
| **Logistic Regression** | **0.843** |
| Support Vector Machine | 0.833 |
| Random Forest | 0.832 |
| K-Nearest Neighbors | 0.832 |
| Gradient Boosting | 0.820 |
| XGBoost | 0.818 |
| Decision Tree | 0.770 |

Selected model (Logistic Regression) on the 154-record held-out test set at a
0.5 threshold: accuracy 0.708, precision 0.600, recall 0.500, F1 0.545,
ROC-AUC 0.813. Recall is modest at 0.5, so half of diabetic test records are
missed; lowering the threshold (try the slider in the app) trades more false
alarms for fewer misses. Glucose is by far the most important feature.

### Ideas for next steps

- Document the dataset's source, licence and population.
- Tune hyperparameters (e.g. with `GridSearchCV`) and choose a decision
  threshold for a stated reason (e.g. a minimum recall).
- Check calibration and add tests for the pipeline and prediction function.

## Safe interpretation

This project is for learning the machine-learning workflow. Its dataset,
preprocessing, and evaluation have not been validated for diagnosis, screening,
treatment, or other clinical decisions. A model's output must not be presented
as a medical diagnosis or substituted for advice from a qualified healthcare
professional.

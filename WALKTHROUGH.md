# Diabetes Prediction Model: Step-by-Step Walkthrough

This guide follows the project in the order the data moves: **CSV → cleaning → training → choosing the best model → saved model → web app → prediction**.

Each step tells you **which file and lines to open**, **what the code does in simple words**, and gives a **"Try it" snippet** you can paste into Python to see the step happen. At the end there is a list of **questions people usually ask, with answers**.

> All "Try it" snippets run from the project folder, after `pip install -r requirements.txt`. Start Python with `python`, or open a new Jupyter notebook in the same folder.

---

## Step 0: The big picture (show this first when you explain it)

```
diabetes.csv ──► diabetes_model.py ──► train.py ──► model/diabetes_model.joblib ──► app.py (website)
 (raw data)      (shared "recipe":      (trains 7 models,  model/metrics.json           (loads the saved
                  columns, cleaning,     picks the best,   (the "brain" and its          model, takes user
                  model list)            tests it, saves)   report card)                 input, predicts)
```

| File | Role in one sentence |
|---|---|
| `diabetes.csv` | The data: 769 patient rows, 8 measurements, plus `Outcome` (1 = diabetic, 0 = not). |
| `DiabetesPredictionModel.ipynb` | Your original exploration notebook (EDA). Still there, still works. It's the "lab notebook". |
| `diabetes_model.py` | The **shared recipe**: column names, how to clean the data, which models to try. Used by both training and the app. |
| `train.py` | The **factory**: trains and compares models, tests the winner, and saves it. Run it once: `python train.py`. |
| `model/diabetes_model.joblib` | The **saved brain**: cleaning steps + trained model in one file. |
| `model/metrics.json` | The **report card**: all scores, charts data, comparison table. The app reads it to draw the performance tab. |
| `app.py` | The **shop window**: a Streamlit website. Run with `streamlit run app.py`. |

**Key idea:** training happens **once** (in `train.py`). The app **does not train**. It only loads the saved brain and asks it for predictions, which is why it's fast.

---

## Step 1: Look at the raw data

**Open:** `diabetes.csv` (or run your notebook's first cells).

Every row is one patient. The 8 input columns ("features") are:

| Column | Meaning |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Blood sugar 2 hours after a glucose drink (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin-fold thickness (mm) |
| Insulin | 2-hour serum insulin (µU/mL) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Score for diabetes in the family |
| Age | Years |

`Outcome` is the answer we want to predict (the "target" or "label").

**Try it**
```python
import pandas as pd
df = pd.read_csv("diabetes.csv")
print(df.shape)                       # (769, 9)
print(df.duplicated().sum())          # 1 duplicate row
print(df["Outcome"].value_counts())   # 0: 500, 1: 269 (with the duplicate)
print((df == 0).sum())                # how many zeros per column
```

**What you will notice:** some columns contain zeros that can't be real. Nobody alive has Glucose = 0 or BMI = 0. (Zeros in `Pregnancies` and `Outcome` are normal.) The snippet counts the raw file, so it shows Insulin = 375. After removing the duplicate row there are:

| Column | Zeros |
|---|---|
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 (almost half!) |
| BMI | 11 |

These zeros mean **"not measured"**. Step 3 shows how we deal with them.

---

## Step 2: The shared settings (`diabetes_model.py`, lines 18–37)

**Open:** `diabetes_model.py`

```python
DATA_PATH  = ROOT / "diabetes.csv"
MODEL_PATH = MODEL_DIR / "diabetes_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET   = "Outcome"
FEATURES = ["Pregnancies", "Glucose", ..., "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
RANDOM_STATE = 42
```

- **Why a separate file?** Training and the app must agree on *exactly* the same columns and cleaning. Putting them in one place means they can't drift apart.
- **Why is `Pregnancies` not in `ZERO_AS_MISSING`?** Zero pregnancies is a real, valid value. Zero glucose is not.
- **`RANDOM_STATE = 42`**: fixes the "randomness" (the data split, the random forest, and so on) so you get the **same results every run**. 42 is just a convention.

---

## Step 3: Loading and cleaning (`diabetes_model.py`, lines 40–65)

### 3a. `load_data()` (lines 46–48)
```python
df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
return df[FEATURES], df[TARGET]
```
It reads the CSV, removes the 1 duplicate row (769 → **768** rows), and splits the table into:
- `X` = the 8 input columns (the questions)
- `y` = `Outcome` (the answers)

### 3b. `zeros_to_nan()` (lines 40–43)
```python
X[X == 0] = np.nan
```
This turns every 0 into `NaN` ("missing"), so the model doesn't think "Insulin = 0" is a real, very low value.

### 3c. `build_preprocessor(scale)` (lines 51–65): the cleaning machine
It builds two small assembly lines and runs each column through the right one:

```
Glucose, BloodPressure, SkinThickness, Insulin, BMI:
    0 → NaN  ──►  fill NaN with the column's median  ──►  (scale)

Pregnancies, DiabetesPedigreeFunction, Age:
    fill any NaN with median  ──►  (scale)
```

- **Median** = the middle value. We use it instead of the mean because it isn't pulled around by extreme values. The medians learned from the training data are Glucose 117, BloodPressure 72, SkinThickness 29, Insulin 125 and BMI 32.4.
- **Scaling (`StandardScaler`)** turns every column into "how many standard deviations from average". Without it, Insulin (values in the hundreds) would look "bigger" than DiabetesPedigreeFunction (values around 0.5). Only some models need this, which is what the `scale` flag is for.
- `ColumnTransformer` is the scikit-learn tool that sends different columns to different assembly lines.

**Try it**: see the cleaning happen
```python
from diabetes_model import load_data, build_preprocessor
X, y = load_data()
pre = build_preprocessor(scale=False).fit(X)
print(X.head(3))                                   # raw: notice Insulin = 0
print(pre.transform(X.head(3)).round(1))            # cleaned: Insulin 0 became 125 (the median)
print(pre.get_feature_names_out())                  # column order after cleaning
```

---

## Step 4: The list of models we try (`diabetes_model.py`, lines 68–86)

`candidate_models()` returns 7 models. Each entry is `(name, model, needs_scaling)`:

| Model | In one line | Needs scaling? |
|---|---|---|
| Logistic Regression | Adds up the features with learned weights and turns the total into a probability | Yes |
| Decision Tree | A flowchart of yes/no questions ("Glucose > 127?") | No |
| Random Forest | 300 decision trees that vote | No |
| Gradient Boosting | Trees built one after another, each fixing the previous one's mistakes | No |
| Support Vector Machine | Finds the best boundary between the two groups | Yes |
| K-Nearest Neighbors | "Look at the 15 most similar patients; what did they have?" | Yes |
| XGBoost | A faster, stronger version of gradient boosting | No |

Tree-based models don't care about scale (a question like "Glucose > 127?" works the same either way), so they skip it.

### `build_pipeline()` (lines 89–90): the most important idea in the project
```python
Pipeline([("preprocess", build_preprocessor(scale)), ("model", estimator)])
```
A **Pipeline glues cleaning and model together into one object**. When you call `.fit()` it learns the medians and scaling **and** trains the model. When you call `.predict()` it cleans the new data the **same way**, then predicts. This is why the app can take raw input (with zeros) and get a correct answer.

---

## Step 5: Training and comparing (`train.py`, lines 38–59)

**Open:** `train.py`. Run it with `python train.py`. It takes about 6 seconds.

### 5a. Split the data FIRST (lines 39–42)
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```
- **80% training (614 rows)**: the model learns from these.
- **20% test (154 rows)**: locked in a drawer. The model never sees them until the very end. They act like a final exam with questions the student has never seen.
- `stratify=y` keeps the same ratio of diabetic to non-diabetic (about 35/65) in both parts.

> **Why split before cleaning?** If we computed the medians using *all* the data (like the old notebook did), the test rows would secretly influence training. This is called **data leakage**, and it makes scores look better than they really are. Because the cleaning lives inside the pipeline, it only ever learns from the training rows.

### 5b. Cross-validation to compare models fairly (lines 45–54)
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(build_pipeline(estimator, scale), X_train, y_train, cv=cv, scoring=CV_METRICS)
```
**5-fold cross-validation** cuts the *training* data into 5 pieces. It then trains on 4 pieces, tests on the 5th, and repeats 5 times so every piece gets a turn as the "mini test". We average the 5 scores.

```
Round 1: [TEST][train][train][train][train]
Round 2: [train][TEST][train][train][train]
...
Round 5: [train][train][train][train][TEST]
```
This gives a more reliable score than one single split, **without touching the real test set**.

### 5c. Pick the winner by ROC-AUC (lines 56–59)
Results from `python train.py`:

```
Logistic Regression      ROC-AUC 0.843   ← winner
Support Vector Machine   ROC-AUC 0.833
Random Forest            ROC-AUC 0.832
K-Nearest Neighbors      ROC-AUC 0.832
Gradient Boosting        ROC-AUC 0.820
XGBoost                  ROC-AUC 0.818
Decision Tree            ROC-AUC 0.770
```
(The ± number printed next to each score is how much it varied across the 5 rounds. The top 4 are very close.)

---

## Step 6: The final exam (`train.py`, lines 61–77)

```python
pipeline = build_pipeline(best_estimator, best_scale).fit(X_train, y_train)   # train winner on all 614 rows
proba = pipeline.predict_proba(X_test)[:, 1]                                 # probability of diabetes for each test patient
pred  = (proba >= 0.5).astype(int)                                           # yes/no using a 0.5 cut-off
```

The winner is trained on the full training set and graded **once** on the 154 locked-away test patients:

| Metric | Value | Meaning in plain words |
|---|---|---|
| Accuracy | 70.8% | Got 109 of 154 right |
| Precision | 60.0% | When it says "diabetic", it's right 60% of the time |
| Recall | 50.0% | It catches 50% of the people who really are diabetic |
| F1 | 54.5% | A balance of precision and recall |
| ROC-AUC | 0.813 | If you pick one diabetic and one non-diabetic patient at random, the model gives the diabetic one a higher probability 81% of the time |

**Confusion matrix** (the four kinds of outcome):

|  | Predicted: no | Predicted: yes |
|---|---|---|
| **Actually no** (100) | 82 ✅ true negatives | 18 ❌ false alarms |
| **Actually yes** (54) | 27 ❌ missed | 27 ✅ caught |

### Permutation importance (lines 75–77)
For each feature, we **shuffle** that column in the test set and see how much ROC-AUC drops. A big drop means the model relies on that feature a lot.

| Feature | ROC-AUC drop |
|---|---|
| **Glucose** | **0.168** (by far the most important) |
| BMI | 0.031 |
| Pregnancies | 0.028 |
| DiabetesPedigreeFunction | 0.009 |
| Age, BloodPressure, Insulin, SkinThickness | ~0 |

---

## Step 7: Saving (`train.py`, lines 79–103)

```python
joblib.dump(pipeline, MODEL_PATH)            # model/diabetes_model.joblib
METRICS_PATH.write_text(json.dumps(metrics)) # model/metrics.json
```
- The `.joblib` file contains the **whole pipeline**: the learned medians, the learned scaling and the trained Logistic Regression weights. It's only about 4 KB.
- `metrics.json` holds everything the app's "Model performance" tab draws: the comparison table, test scores, confusion matrix, ROC curve points, every test patient's probability (for the threshold slider), feature importance and the medians.

**Try it**: open the saved brain and look inside
```python
from diabetes_model import load_model
model = load_model()
print(model)                                          # the full pipeline
lr  = model.named_steps["model"]
pre = model.named_steps["preprocess"]
for name, w in zip(pre.get_feature_names_out(), lr.coef_[0]):
    print(f"{name:40s} {w:+.3f}")
```
You will see these weights (larger = pushes more toward "diabetic"):

| Feature (scaled) | Weight |
|---|---|
| Glucose | **+1.183** |
| BMI | +0.689 |
| Pregnancies | +0.377 |
| DiabetesPedigreeFunction | +0.233 |
| Age | +0.148 |
| SkinThickness | +0.028 |
| BloodPressure | −0.044 |
| Insulin | −0.066 |
| *(intercept)* | −0.872 |

---

## Step 8: How ONE prediction is made (do this by hand once; it removes all the "magic")

`predict_proba()` in `diabetes_model.py` (lines 97–100) turns a dictionary into a one-row table and asks the pipeline for the probability.

Let's follow the **"higher-risk example"** patient from the app:

```
Pregnancies 6, Glucose 148, BloodPressure 72, SkinThickness 35,
Insulin 0, BMI 33.6, DiabetesPedigreeFunction 0.627, Age 50
```

1. **Clean**: Insulin 0 → NaN → filled with 125 (the median).
2. **Scale**: each value becomes "standard deviations from average". For example, Glucose 148 → **+0.878** and Age 50 → **+1.407**.
3. **Score** = intercept + Σ (weight × scaled value) = **0.875**
4. **Probability** = sigmoid(score) = 1 / (1 + e^(−0.875)) = **0.706 → 70.6%**
5. **Decision**: 70.6% ≥ 50% → **"likely diabetic"**

**Try it**: reproduce the 70.6% yourself
```python
import numpy as np, pandas as pd
from diabetes_model import load_model, predict_proba
model = load_model()
patient = dict(Pregnancies=6, Glucose=148, BloodPressure=72, SkinThickness=35,
               Insulin=0, BMI=33.6, DiabetesPedigreeFunction=0.627, Age=50)

print(predict_proba(model, patient))                        # 0.7058...

pre, lr = model.named_steps["preprocess"], model.named_steps["model"]
scaled = pre.transform(pd.DataFrame([patient]))[0]          # steps 1-2
score  = scaled @ lr.coef_[0] + lr.intercept_[0]            # step 3
print(score, 1 / (1 + np.exp(-score)))                      # 0.875  0.7058  (same number!)
```

---

## Step 9: The web app (`app.py`)

**Run:** `streamlit run app.py` and open http://localhost:8501

**How Streamlit works (one sentence):** the whole `app.py` script re-runs from top to bottom every time the user changes something, and Streamlit draws whatever the script outputs.

### 9a. Startup (lines 37–52)
```python
@st.cache_resource
def get_model():
    if not MODEL_PATH.exists() ...: train.main()   # safety net: train if the model is missing
    return load_model(), json.loads(METRICS_PATH.read_text())
```
`@st.cache_resource` / `@st.cache_data` mean "do this once and remember it". Without them, the model would be reloaded on every click.

### 9b. Tab 1: "Try the model" (lines 64–139)
1. **Presets** (lines 66–75): "lower-risk" and "higher-risk" are just real rows from the CSV (one with Outcome 0 and one with Outcome 1) copied into the input boxes.
2. **Inputs** (lines 77–89): 8 number boxes. Their limits and help text come from the `FIELD_INFO` dictionary at the top (lines 25–34).
3. **Prediction** (line 91): `proba = predict_proba(model, record)`. This is exactly Step 8.
4. **Gauge and colored box** (lines 93–117): green if below 50%, red if 50% or above.
5. **"Why?" chart** (lines 119–137): for each feature, it replaces just that one value with a "typical" (median) value, predicts again, and shows the difference:
   ```python
   baseline = dict(record, **{f: medians[f]})            # same patient, but this feature set to typical
   effect   = proba - predict_proba(model, baseline)      # how much this feature moved the probability
   ```
   A red bar means "this value raised the risk compared with a typical person". A green bar means it lowered it.

### 9c. Tab 2: "Model performance" (lines 141–215)
Everything here is read from `metrics.json`. **Nothing is re-trained.**
- The metric cards, confusion matrix and ROC curve show the Step 6 results.
- **Threshold slider** (lines 171–186): uses the saved test-set probabilities and simply re-counts how many would be right or wrong at the new cut-off:
  ```python
  pred = (p >= t).astype(int)       # same as train.py line 64, but with t instead of 0.5
  ```
- The model comparison and feature importance charts show Steps 5c and 6.

### 9d. Tab 3: "Explore the data" (lines 217–238)
Shows class counts and a histogram of any feature, split into diabetic vs non-diabetic (great for showing that diabetic patients tend to have higher Glucose).

### 9e. Tab 4: "How it works" (lines 240–266)
A plain-text summary of this pipeline, for visitors.

---

## Step 10: A suggested 5-minute demo script

1. **"Explore the data" tab** → pick **Glucose**. "Diabetic patients (red) clearly have higher glucose. The model learns patterns like this."
2. **"Try the model" tab** → click **higher-risk example** → 70.6%, red. Point to the "Why?" chart: Glucose is the biggest red bar.
3. Change **Glucose to 100**. The probability drops a lot. "The model reacts in a sensible direction."
4. Click **lower-risk example** → 3.7%, green.
5. **"Model performance" tab** → "We tested on 154 patients the model never saw: 71% accuracy, AUC 0.81."
6. Move the **threshold slider to 0.30** → recall goes from 50% to 80% and missed patients drop from 27 to 11. "For a screening tool you'd rather have a false alarm than miss someone, so you might lower the threshold."
7. Finish with: "This is a learning project. It's **not** a medical diagnosis."

---

## Questions people usually ask, and answers

**Q: Why Logistic Regression and not something "fancier" like XGBoost?**
We let the data decide: all 7 models were compared fairly with cross-validation, and Logistic Regression scored best (0.843 AUC). On a small dataset (768 rows) with simple numeric features, simple models often beat complex ones, which can **overfit** (memorize the training data). It's also easy to explain, because every feature has one weight (Step 7).

**Q: Why is cross-validation accuracy (78.8%) higher than test accuracy (70.8%)?**
With only 154 test patients, a few lucky or unlucky cases move the score by several percent. The test set is the honest final number. The CV numbers were used **only to choose** the model. That's why we never pick a model based on the test score.

**Q: What is ROC-AUC and why use it instead of accuracy?**
Accuracy depends on the 0.5 cut-off and can be misleading with unbalanced classes: a model that always says "not diabetic" scores 65% accuracy while being useless. ROC-AUC measures how well the model **ranks** diabetic patients above non-diabetic ones across *all* thresholds. 0.5 is random guessing and 1.0 is perfect.

**Q: Recall is only 50%. Isn't that bad?**
At the 0.5 cut-off, yes, it misses half of the diabetic patients. But 0.5 is just a default. Lowering the threshold to 0.3 catches 80% (with more false alarms). Choosing the threshold is a **decision about costs** (missing a patient vs. a false alarm), not a property of the model. The slider in the app shows this trade-off live.

**Q: Why replace zeros with the median? Isn't that making up data?**
A zero there means "not measured". Leaving it as 0 would teach the model something false ("Insulin 0 = very low insulin"). Filling with the median is a simple, common, neutral choice: it means "assume a typical value". For Insulin, almost half the values are missing, which is partly why Insulin ends up with almost no importance.

**Q: What is "data leakage" and how did you avoid it?**
Leakage means information from the test set sneaks into training, which makes the scores look too good. The old notebook computed medians on *all* rows before splitting. Now the cleaning is inside the `Pipeline`, so the medians and scaling are learned from the training rows only (Step 5a).

**Q: Does the app re-train the model every time?**
No. `train.py` trains once and saves `model/diabetes_model.joblib`. The app only loads it (Step 9a). If you change the data or the models, run `python train.py` again.

**Q: Where does the "probability" come from?**
Logistic Regression computes a score (weighted sum) and passes it through the sigmoid function, which squeezes any number into the range 0–1. You can do it by hand (Step 8).

**Q: What does the "Why?" chart actually show?**
For each input, how much the probability would change if that one value were replaced with a typical value from the training data. It's a simple "what-if" explanation. For Insulin = 0, the effect is 0, because 0 is treated as missing and already filled with the typical value.

**Q: Why does Insulin have a negative weight? Shouldn't high insulin mean diabetes?**
Its weight (−0.066) is close to zero, which means "almost no effect once Glucose and BMI are known". Features are correlated (Insulin moves with Glucose), so Glucose "takes the credit". Also, half of the Insulin values were missing. Don't over-read small weights.

**Q: How do I share it with someone?**
Deploy free on Streamlit Community Cloud: sign in with GitHub → Create app → pick this repo, branch `main`, file `app.py` → Deploy. You get a public link (also described in the README).

**Q: Can this be used by doctors?**
No. It's trained on 768 records from one specific population (Pima Indian women, age 21+), it hasn't been clinically validated, and recall at the default threshold is 50%. It's a learning project showing how an ML workflow works.

**Q: How would you improve it?**
Tune each model's settings (e.g. with `GridSearchCV`), choose the threshold for a stated reason (e.g. "at least 80% recall"), try better ways of filling missing values, collect more data, and check that the probabilities are well calibrated.

---

## Glossary

| Term | Simple meaning |
|---|---|
| Feature | An input column (Glucose, Age, …) |
| Target / label | The answer we predict (`Outcome`) |
| Training set | Data the model learns from (80%) |
| Test set | Data kept hidden for the final exam (20%) |
| Cross-validation | Repeated train/test on pieces of the training set to compare models fairly |
| Pipeline | Cleaning + model glued together so they always run in the same order |
| Imputation | Filling in missing values (here, with the median) |
| Scaling | Putting all features on a similar scale |
| Overfitting | Memorizing the training data instead of learning general patterns |
| Threshold | The probability cut-off that turns "70%" into "yes" |
| Precision | Of the people flagged "diabetic", how many really are |
| Recall | Of the people who really are diabetic, how many we flagged |
| ROC-AUC | How well the model ranks positives above negatives (0.5 = random, 1 = perfect) |
| Data leakage | Test information sneaking into training, which inflates the scores |

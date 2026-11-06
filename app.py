"""Interactive demo of the diabetes prediction model.

Run locally:  streamlit run app.py
"""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from diabetes_model import (
    DATA_PATH,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    TARGET,
    load_model,
    predict_proba,
)

st.set_page_config(page_title="Diabetes Prediction Model", page_icon="🩺", layout="wide")

FIELD_INFO = {
    "Pregnancies": dict(label="Pregnancies", min=0, max=20, step=1, default=3, help="Number of times pregnant"),
    "Glucose": dict(label="Glucose (mg/dL)", min=0, max=250, step=1, default=120, help="Plasma glucose concentration, 2 hours into an oral glucose tolerance test. 0 = not measured."),
    "BloodPressure": dict(label="Blood pressure (mm Hg)", min=0, max=140, step=1, default=70, help="Diastolic blood pressure. 0 = not measured."),
    "SkinThickness": dict(label="Skin thickness (mm)", min=0, max=100, step=1, default=29, help="Triceps skin-fold thickness. 0 = not measured."),
    "Insulin": dict(label="Insulin (µU/mL)", min=0, max=900, step=1, default=125, help="2-hour serum insulin. 0 = not measured."),
    "BMI": dict(label="BMI (kg/m²)", min=0.0, max=70.0, step=0.1, default=32.0, help="Body mass index. 0 = not measured."),
    "DiabetesPedigreeFunction": dict(label="Diabetes pedigree function", min=0.0, max=2.5, step=0.01, default=0.37, help="Score summarising diabetes history in relatives"),
    "Age": dict(label="Age (years)", min=21, max=90, step=1, default=33, help="Age in years (dataset covers ages 21+)"),
}


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        import train

        train.main()
    return load_model(), json.loads(METRICS_PATH.read_text())


@st.cache_data
def get_data():
    return pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)


model, metrics = get_model()
data = get_data()

st.title("🩺 Diabetes Prediction Model")
st.caption(
    f"A machine-learning demo trained on the Pima Indians diabetes dataset ({len(data)} records). "
    f"Model in use: **{metrics['best_model']}** — chosen by 5-fold cross-validated ROC-AUC."
)

tab_predict, tab_perf, tab_data, tab_how = st.tabs(
    ["🔮 Try the model", "📊 Model performance", "🗂️ Explore the data", "⚙️ How it works"]
)

# --------------------------------------------------------------------------- Predict
with tab_predict:
    presets = {
        "Custom": None,
        "Example: lower-risk profile": data[data[TARGET] == 0].iloc[1][FEATURES].to_dict(),
        "Example: higher-risk profile": data[data[TARGET] == 1].iloc[0][FEATURES].to_dict(),
    }
    preset = st.radio("Start from", list(presets), horizontal=True)
    if presets[preset] is not None and st.session_state.get("_preset") != preset:
        for f, v in presets[preset].items():
            st.session_state[f] = type(FIELD_INFO[f]["default"])(v)
    st.session_state["_preset"] = preset

    left, right = st.columns([1, 1], gap="large")
    record = {}
    with left:
        st.subheader("Patient measurements")
        c1, c2 = st.columns(2)
        for i, f in enumerate(FEATURES):
            info = FIELD_INFO[f]
            st.session_state.setdefault(f, info["default"])
            with (c1 if i % 2 == 0 else c2):
                record[f] = st.number_input(
                    info["label"], min_value=info["min"], max_value=info["max"],
                    step=info["step"], help=info["help"], key=f,
                )

    proba = predict_proba(model, record)
    threshold = metrics["threshold"]
    with right:
        st.subheader("Prediction")
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                title={"text": "Estimated probability of diabetes"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#d62728" if proba >= threshold else "#2ca02c"},
                    "steps": [
                        {"range": [0, threshold * 100], "color": "#e8f5e9"},
                        {"range": [threshold * 100, 100], "color": "#fdecea"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100},
                },
            )
        )
        gauge.update_layout(height=280, margin=dict(t=60, b=10, l=30, r=30))
        st.plotly_chart(gauge, width="stretch")
        if proba >= threshold:
            st.error(f"**Model output: likely diabetic** (probability ≥ {threshold:.0%} decision threshold)")
        else:
            st.success(f"**Model output: likely not diabetic** (probability < {threshold:.0%} decision threshold)")

        # What-if explanation: how much does each input move the prediction away from a "typical" value?
        medians = metrics["training_medians"]
        effects = []
        for f in FEATURES:
            baseline = dict(record, **{f: medians[f]})
            effects.append({"feature": FIELD_INFO[f]["label"], "effect": (proba - predict_proba(model, baseline)) * 100})
        effects = pd.DataFrame(effects).sort_values("effect")
        fig = px.bar(
            effects, x="effect", y="feature", orientation="h",
            color=np.where(effects["effect"] >= 0, "Raises risk", "Lowers risk"),
            color_discrete_map={"Raises risk": "#d62728", "Lowers risk": "#2ca02c"},
            labels={"effect": "Change in probability (percentage points)", "feature": "", "color": ""},
            title="Why? Effect of each input vs. a typical (median) value",
        )
        fig.update_layout(height=340, margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

    st.info(
        "⚠️ Educational demo only. This model was trained on a small, specific dataset and is **not** a "
        "medical diagnosis. Always consult a qualified healthcare professional."
    )

# --------------------------------------------------------------------------- Performance
with tab_perf:
    tm = metrics["test_metrics"]
    st.subheader(f"Held-out test set results ({metrics['n_test']} patients the model never saw)")
    cols = st.columns(5)
    for col, (name, key) in zip(
        cols,
        [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1 score", "f1"), ("ROC-AUC", "roc_auc")],
    ):
        col.metric(name, f"{tm[key]:.1%}" if key != "roc_auc" else f"{tm[key]:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        cm = np.array(metrics["confusion_matrix"])
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=["Predicted: no diabetes", "Predicted: diabetes"],
            y=["Actual: no diabetes", "Actual: diabetes"],
            title="Confusion matrix (threshold 0.5)",
        )
        fig.update_layout(coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, width="stretch")
    with c2:
        roc = metrics["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"Model (AUC {tm['roc_auc']:.3f})", fill="tozeroy"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(dash="dash", color="grey")))
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate (recall)", height=380)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Try a different decision threshold")
    st.caption(
        "The model outputs a probability; the threshold turns it into a yes/no answer. Lowering it catches more "
        "diabetic patients (higher recall) at the cost of more false alarms (lower precision)."
    )
    t = st.slider("Threshold", 0.05, 0.95, 0.5, 0.05)
    tp_data = metrics["test_predictions"]
    p, y = np.array(tp_data["proba"]), np.array(tp_data["label"])
    pred = (p >= t).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{(tp + tn) / len(y):.1%}")
    cols[1].metric("Precision", f"{tp / (tp + fp):.1%}" if tp + fp else "n/a")
    cols[2].metric("Recall", f"{tp / (tp + fn):.1%}" if tp + fn else "n/a")
    cols[3].metric("Missed diabetic patients", fn)

    c1, c2 = st.columns(2)
    with c1:
        comp = pd.DataFrame(metrics["cv_comparison"])
        fig = px.bar(
            comp.sort_values("roc_auc"), x="roc_auc", y="model", orientation="h", error_x="roc_auc_std",
            text=comp.sort_values("roc_auc")["roc_auc"].map("{:.3f}".format),
            labels={"roc_auc": "Mean ROC-AUC (5-fold CV)", "model": ""},
            title="Model comparison (cross-validation on training data)",
        )
        fig.update_layout(height=400, xaxis_range=[0.5, 1])
        st.plotly_chart(fig, width="stretch")
    with c2:
        fi = pd.Series(metrics["feature_importance"]).sort_values()
        fig = px.bar(
            x=fi.values, y=[FIELD_INFO[f]["label"] for f in fi.index], orientation="h",
            labels={"x": "Drop in ROC-AUC when feature is shuffled", "y": ""},
            title="Which features matter most (permutation importance)",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

    with st.expander("Full cross-validation table"):
        st.dataframe(
            comp[["model", "roc_auc", "accuracy", "precision", "recall", "f1"]].style.format(
                {c: "{:.3f}" for c in ["roc_auc", "accuracy", "precision", "recall", "f1"]}
            ),
            hide_index=True, width="stretch",
        )

# --------------------------------------------------------------------------- Data
with tab_data:
    counts = data[TARGET].value_counts().sort_index()
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", len(data))
    c2.metric("Diabetic (Outcome = 1)", f"{counts[1]} ({counts[1] / len(data):.0%})")
    c3.metric("Non-diabetic (Outcome = 0)", f"{counts[0]} ({counts[0] / len(data):.0%})")

    feature = st.selectbox("Compare a feature between the two groups", FEATURES, index=1, format_func=lambda f: FIELD_INFO[f]["label"])
    plot_df = data.assign(Group=data[TARGET].map({0: "Non-diabetic", 1: "Diabetic"}))
    fig = px.histogram(
        plot_df, x=feature, color="Group", barmode="overlay", marginal="box", nbins=40,
        color_discrete_map={"Non-diabetic": "#2ca02c", "Diabetic": "#d62728"},
        labels={feature: FIELD_INFO[feature]["label"]},
    )
    st.plotly_chart(fig, width="stretch")
    zero_count = int((data[feature] == 0).sum())
    if feature in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"] and zero_count:
        st.caption(f"{zero_count} records have {feature} = 0, which means *not measured*; the model treats these as missing.")

    with st.expander("Raw data"):
        st.dataframe(data, width="stretch")

# --------------------------------------------------------------------------- How it works
with tab_how:
    st.markdown(
        f"""
### Pipeline
1. **Load data** – `diabetes.csv`, exact duplicate rows removed ({len(data)} records left).
2. **Split** – 80 % training / 20 % test, stratified so both keep the same diabetic / non-diabetic ratio.
   The test set is locked away until the very end.
3. **Preprocess (inside a scikit-learn `Pipeline`)**
   - In *Glucose, BloodPressure, SkinThickness, Insulin, BMI* a value of **0 means "not measured"**, so zeros
     are turned into missing values and filled with the **median of the training data**.
   - Features are standardised (mean 0, std 1) for models that need it (Logistic Regression, SVM, KNN).
   - Because this happens inside the pipeline, the test data never leaks into the preprocessing.
4. **Compare 7 models** with 5-fold cross-validation on the training set, ranked by ROC-AUC.
5. **Final check** – the best model (**{metrics['best_model']}**) is retrained on all training data and scored
   **once** on the held-out test set.
6. **Save** – the whole pipeline (preprocessing + model) is saved to `model/diabetes_model.joblib`, so the
   app applies exactly the same steps to your inputs.

### Reproduce
```bash
pip install -r requirements.txt
python train.py       # retrain and save the model + metrics
streamlit run app.py  # launch this app
```
"""
    )

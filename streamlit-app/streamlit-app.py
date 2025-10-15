# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from utils.loader import load_dataset
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="ASD Screening Dashboard", layout="wide")
st.title("🧩 Autism Spectrum Disorder (ASD) Screening Dashboard")

# Load dataset
df = load_dataset("data/autism_data.csv")
st.sidebar.header("Filter options")
age_range = st.sidebar.slider("Age range", int(df.age.min()), int(df.age.max()),
                              (int(df.age.min()), int(df.age.max())))
gender_filter = st.sidebar.multiselect("Gender", df['gender'].unique().tolist(),
                                       default=df['gender'].unique().tolist())

filtered = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])
              & (df['gender'].isin(gender_filter))]

st.write("### Dataset Preview")
st.dataframe(filtered.head())

# Model training
st.subheader("⚙️ Train / Evaluate Model")
if st.button("Train Model"):
    st.info("Training model... please wait.")
    X = pd.get_dummies(filtered.drop(columns=['Class/ASD']))
    y = filtered['Class/ASD']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=0.2,
                                                        random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    st.success(f"Model trained successfully! AUC = {auc:.3f}")
    st.text(classification_report(y_test, y_pred))
    joblib.dump({"model": model, "features": X.columns.tolist()},
                "models/baseline_rf.joblib")

# Prediction section
st.subheader("🔍 Single Prediction")
if "baseline_rf.joblib" in [f.name for f in st.runtime.exists_dir("models")]:
    data = joblib.load("models/baseline_rf.joblib")
    model, features = data["model"], data["features"]

    st.write("Enter person details:")
    age = st.number_input("Age", 0, 120, 10)
    gender = st.selectbox("Gender", df['gender'].unique())
    jaundice = st.selectbox("Jaundice at birth?", df['jaundice'].unique())
    family_history = st.selectbox("Family history of ASD?", df['family_history'].unique())

    if st.button("Predict"):
        input_data = pd.DataFrame([{
            "age": age,
            "gender_" + gender: 1,
            "jaundice_" + jaundice: 1,
            "family_history_" + family_history: 1
        }])
        input_data = input_data.reindex(columns=features, fill_value=0)
        prob = model.predict_proba(input_data)[:, 1][0]
        st.metric("Predicted ASD Probability", f"{prob:.2f}")

        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        st.write("### 🔎 Feature Contribution (SHAP)")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
        st.pyplot(fig)
else:
    st.warning("Train the model first to enable predictions.")

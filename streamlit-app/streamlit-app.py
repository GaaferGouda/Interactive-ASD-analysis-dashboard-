import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Autism Behavior Dashboard", layout="wide")

# ---------------------- Data Loading ----------------------
@st.cache_data
def load_dataset():
    """Load Autism dataset from local CSV or sample data."""
    try:
        df = pd.read_csv("data/autism_data.csv")
    except FileNotFoundError:
        # Fallback sample dataset
        df = pd.DataFrame({
            "age": [3, 4, 5, 6, 7],
            "gender": ["M", "F", "M", "F", "M"],
            "A1_Score": [1, 0, 1, 1, 0],
            "A2_Score": [1, 1, 0, 1, 1],
            "A3_Score": [0, 1, 1, 0, 1],
            "autism": [1, 0, 1, 0, 1]
        })
    return df

df = load_dataset()

# ---------------------- Sidebar ----------------------
st.sidebar.title("⚙️ Dashboard Controls")
show_data = st.sidebar.checkbox("Show Raw Data", True)
run_model = st.sidebar.button("Run Prediction Model")

# ---------------------- Data Display ----------------------
st.title("🧩 Autism Behavior Monitoring Dashboard")

if show_data:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df)

# ---------------------- Basic Analysis ----------------------
st.subheader("📈 Basic Statistics")
st.write(df.describe())

# ---------------------- Visualization ----------------------
st.subheader("🔍 Data Visualization")
fig, ax = plt.subplots()
sns.countplot(x="gender", hue="autism", data=df, ax=ax)
st.pyplot(fig)

# ---------------------- Machine Learning ----------------------
if run_model:
    st.subheader("🤖 Prediction Model Results")

    X = df.drop(columns=["autism"])
    X = pd.get_dummies(X, drop_first=True)
    y = df["autism"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.metric("Model Accuracy", f"{acc:.2f}")
    st.write("Confusion Matrix:")
    st.write(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]))

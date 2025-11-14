# ============================================================
# 🏥 SMART HEALTHCARE STREAMLIT APP (INTERACTIVE DASHBOARD)
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="Smart Survey Analysis", layout="wide")
st.title("🏥 Smart Survey Analysis platform")
st.markdown("""
This dashboard provides insights based on hospital survey data.
Use the filters below to explore patient demographics, test outcomes, and survival predictions.
""")

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    data_path = os.path.join(os.getcwd(), "healthcare_data.csv")
    if not os.path.exists(data_path):
        st.error("❌ CSV file 'healthcare_data.csv' not found in project folder!")
        st.stop()
    df = pd.read_csv(data_path)

    # Convert dates to datetime and compute Length_of_Stay
    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
    df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])
    df["Length_of_Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
    return df

df = load_data()

# ------------------------------
# Load Survival Classifier
# ------------------------------
@st.cache_resource
def load_survival_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "survival_classifier.joblib")

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()

    clf_model = joblib.load(model_path)
    return clf_model

survival_model = load_survival_model()

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.header("🔍 Filter Data")
selected_gender = st.sidebar.multiselect("Gender", df["Gender"].unique(), default=df["Gender"].unique())
selected_blood = st.sidebar.multiselect("Blood Type", df["Blood Type"].unique(), default=df["Blood Type"].unique())
selected_med = st.sidebar.multiselect("Medication", df["Medication"].unique(), default=df["Medication"].unique())
selected_test = st.sidebar.multiselect("Test Results", df["Test Results"].unique(), default=df["Test Results"].unique())

filtered_df = df[
    (df["Gender"].isin(selected_gender)) &
    (df["Blood Type"].isin(selected_blood)) &
    (df["Medication"].isin(selected_med)) &
    (df["Test Results"].isin(selected_test))
]

st.write("### 🧾 Filtered Data Overview")
st.dataframe(filtered_df.head(10))

# ------------------------------
# Dashboard: Interactive Charts
# ------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Age Distribution")
    fig_age = px.histogram(
        filtered_df,
        x="Age",
        nbins=20,
        color="Gender",
        title="Age Distribution by Gender",
        marginal="box"
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    st.subheader("🧬 Gender Ratio")
    fig_gender = px.pie(filtered_df, names="Gender", title="Gender Proportion", hole=0.3)
    st.plotly_chart(fig_gender, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.subheader("💉 Blood Type Distribution")
    blood_counts = filtered_df["Blood Type"].value_counts().reset_index()
    blood_counts.columns = ["Blood Type", "Count"]
    fig_blood = px.bar(
        blood_counts,
        x="Blood Type",
        y="Count",
        text="Count",
        title="Count of Each Blood Group"
    )
    st.plotly_chart(fig_blood, use_container_width=True)

with col4:
    st.subheader("💊 Top Prescribed Medications")
    med_counts = filtered_df["Medication"].value_counts().reset_index()
    med_counts.columns = ["Medication", "Count"]
    fig_med = px.bar(
        med_counts.head(10),
        x="Medication",
        y="Count",
        text="Count",
        title="Top 10 Medications"
    )
    st.plotly_chart(fig_med, use_container_width=True)

st.subheader("🧪 Test Results Overview")
test_counts = filtered_df["Test Results"].value_counts().reset_index()
test_counts.columns = ["Test Results", "Count"]
fig_test = px.pie(
    test_counts,
    names="Test Results",
    values="Count",
    title="Test Outcome Proportions",
    hole=0.3
)
st.plotly_chart(fig_test, use_container_width=True)

# ------------------------------
# Prediction Section (Survival)
# ------------------------------
st.markdown("---")
st.header("🔎 Predict Patient Survival")

st.write("Enter patient details to predict whether the patient will survive:")

age = st.number_input("Age", min_value=0, max_value=120, value=35)
gender = st.selectbox("Gender", df["Gender"].unique())
blood = st.selectbox("Blood Type", df["Blood Type"].unique())
med = st.selectbox("Medication", df["Medication"].unique())
test = st.selectbox("Test Results", df["Test Results"].unique())
condition = st.selectbox("Medical Condition", df["Medical Condition"].unique())
admission_type = st.selectbox("Admission Type", df["Admission Type"].unique())

# Prepare input DataFrame for prediction
input_df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Blood Type": [blood],
    "Medical Condition": [condition],
    "Medication": [med],
    "Test Results": [test],
    "Admission Type": [admission_type],
    "Doctor": [df["Doctor"].mode()[0]],
    "Hospital": [df["Hospital"].mode()[0]],
    "Insurance Provider": [df["Insurance Provider"].mode()[0]],
    "Room Number": [df["Room Number"].mode()[0]],
    "Length_of_Stay": [df["Length_of_Stay"].mean()]
})

if st.button("▫️ Predict Survival"):
    try:
        pred = survival_model.predict(input_df)[0]
        result = "✅ Yes" if pred == 1 else "❌ No"
        st.success(f"Will the patient survive? {result}")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

st.markdown("---")
st.caption("© 2025 Smart Healthcare Analytics | Built with Streamlit ")

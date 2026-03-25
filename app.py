import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# Title
# =========================
st.title("🏠 Loan Prediction App (KNN)")

# =========================
# Load Dataset (Directly)
# =========================
df = pd.read_csv("knn_regression_dataset.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# =========================
# Handle Missing Values
# =========================
df.fillna(df.mean(numeric_only=True), inplace=True)

if "date" in df.columns:
    df.drop("date", axis=1, inplace=True)

# =========================
# Encoding
# =========================
categorical_cols = df.select_dtypes(include="object").columns
df = pd.get_dummies(df, columns=categorical_cols)

# =========================
# Check Target
# =========================
if "target" not in df.columns:
    st.error("Dataset must contain 'target' column")
    st.stop()

# =========================
# Split Features & Target
# =========================
X = df.drop("target", axis=1)
y = df["target"]

# =========================
# Scaling (ONLY X)
# =========================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# =========================
# Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Select K
# =========================
k = st.slider("Select K value", 1, 20, 5)

# =========================
# Train Model
# =========================
model = KNeighborsRegressor(n_neighbors=k)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# =========================
# Metrics
# =========================
st.subheader("📊 Model Performance")
st.write(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
st.write(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# =========================
# Prediction Section
# =========================
st.subheader("🔮 Make Prediction")

# Select boxes
city = st.selectbox("City", ["Hyderabad", "Chennai", "Bangalore", "Mumbai"])
employment_type = st.selectbox(
    "Employment Type", ["Salaried", "Self-Employed", "Unemployed"]
)
loan_type = st.selectbox("Loan Type", ["Home", "Personal"])

# Numeric inputs
age = st.number_input("Age", value=30)
income = st.number_input("Income", value=50000)
loan_amount = st.number_input("Loan Amount", value=200000)
credit_score = st.number_input("Credit Score", value=700)

# Convert input
input_dict = {
    "age": age,
    "income": income,
    "loan_amount": loan_amount,
    "credit_score": credit_score,
}

input_df = pd.DataFrame([input_dict])

# Manual encoding
input_df["city_Bangalore"] = 1 if city == "Bangalore" else 0
input_df["city_Chennai"] = 1 if city == "Chennai" else 0
input_df["city_Hyderabad"] = 1 if city == "Hyderabad" else 0
input_df["city_Mumbai"] = 1 if city == "Mumbai" else 0

input_df["employment_type_Salaried"] = 1 if employment_type == "Salaried" else 0
input_df["employment_type_Self-Employed"] = 1 if employment_type == "Self-Employed" else 0
input_df["employment_type_Unemployed"] = 1 if employment_type == "Unemployed" else 0

input_df["loan_type_Home"] = 1 if loan_type == "Home" else 0
input_df["loan_type_Personal"] = 1 if loan_type == "Personal" else 0

# Match columns
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Value: {prediction[0]:.4f}")
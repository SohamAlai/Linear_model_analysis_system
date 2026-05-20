import sys
import os
sys.path.append(os.path.abspath('..'))
from utils.preprocessing import get_columns_types,create_preprocessor


import streamlit as st

st.title("Linear Models Regression Lab")

import pandas as pd

df = pd.read_csv("data/housing.csv")
df = df.dropna()
model_name = st.sidebar.selectbox(
    "Select Model",
    ["Linear", "Ridge", "Lasso", "ElasticNet"]
)

alpha = st.sidebar.slider("Alpha", 0.01, 10.0, 1.0)

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]


num_cols, cat_cols = get_columns_types(X)
preprocessor = create_preprocessor(num_cols, cat_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
if model_name == "Linear":
    alpha = None

if model_name == "Linear":
    model = LinearRegression()

elif model_name == "Ridge":
    model = Ridge(alpha=alpha)

elif model_name == "Lasso":
    model = Lasso(alpha=alpha)

elif model_name == "ElasticNet":
    model = ElasticNet(alpha=alpha)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

st.write("### Model Performance")
st.write(f"R2 Score: {r2:.4f}")
st.write(f"MAE: {mae:.2f}")
st.write(f"MSE: {mse:.2f}")


st.write("## 🔮 Make a Prediction")

longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=41.0)
total_rooms = st.number_input("Total Rooms", value=880.0)
total_bedrooms = st.number_input("Total Bedrooms", value=129.0)
population = st.number_input("Population", value=322.0)
households = st.number_input("Households", value=126.0)
median_income = st.number_input("Median Income", value=8.3252)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

import pandas as pd

input_data = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [ocean_proximity]
})

if st.button("Predict"):
    prediction = pipeline.predict(input_data)
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")



import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.scatter(y_test, preds)
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title("Actual vs Predicted")

st.pyplot(fig)

residuals = y_test - preds

fig2, ax2 = plt.subplots()
ax2.scatter(preds, residuals)
ax2.axhline(y=0)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Residuals")
ax2.set_title("Residual Plot")

st.pyplot(fig2)


st.title("🏠 Housing Price Prediction App")

st.sidebar.header("⚙️ Model Settings")

st.header("📊 Model Performance")

st.header("🔮 Prediction")

st.write("""
This app compares Linear Regression models (Linear, Ridge, Lasso, ElasticNet)
and predicts housing prices based on input features.
""")

coef = pipeline.named_steps["model"].coef_

feature_names = num_cols + list(
    preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
)

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coef
})

st.write("### Feature Importance")
st.dataframe(coef_df.sort_values(by="Coefficient", ascending=False))
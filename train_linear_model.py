import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge # Using Ridge to handle multicollinearity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load dataset
print("Loading dataset...")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "immo_data.csv")

try:
    df = pd.read_csv(data_path)
except:
    df = pd.read_csv("data/immo_data.csv")


# 2. Aggressive Cleaning (The "Recruiter-Ready" Logic)
# Focus on valid data only
df = df[(df["livingSpace"] > 20) & (df["livingSpace"] < 250)]
df = df[(df["baseRent"] > 200) & (df["baseRent"] < 5000)]

# Create target
df["rent_per_sqm"] = df["baseRent"] / df["livingSpace"]

# REMOVE OUTLIERS: Keep only the middle 95% of the market. 
# Values like 1€ or 100€/sqm are errors in the immo_data dataset.
df = df[(df["rent_per_sqm"] >= 7) & (df["rent_per_sqm"] <= 35)]
df = df[(df["yearConstructed"] > 1900) & (df["yearConstructed"] <= 2026)]


# 3. Feature selection
numeric_features = ["livingSpace", "yearConstructed", "noRooms", "floor"]
categorical_features = ["condition", "interiorQual"]
binary_features = ["lift", "balcony", "newlyConst"]

features = numeric_features + categorical_features + binary_features
target = "rent_per_sqm"

df = df[features + [target]].dropna()

# 4. Train-test split
X = df[features]
y = np.log1p(df[target]) # LOG TRANSFORMATION to fix the skew

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Preprocessing (Standardization is Key)
numeric_transformer = Pipeline(steps=[
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()) # Scaler MUST be after Poly to normalize high-degree terms
])

categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop="first")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("bin", "passthrough", binary_features)
    ]
)

# 6. Ridge Regression (Better for High-Dimensional Data)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=1.0)) # Ridge is more stable than LinearRegression
])

# 7. Train
print(f"Training on {len(X_train)} cleaned samples...")
model.fit(X_train, y_train)


# 8. Evaluate (Convert back from Log Scale)
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)

mae = mean_absolute_error(y_test_real, y_pred)
r2 = r2_score(y_test_real, y_pred)

print("\n" + "="*40)
print("UPDATED MODEL RESULTS")
print("="*40)
print(f"Mean Absolute Error: {mae:.2f} €/m²")
print(f"R² Score: {r2:.4f}") 
print("="*40)

# 9. Save model
model_dir = os.path.join(script_dir, "models")
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "linear_rent_model.pkl"))
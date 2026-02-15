import os
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data & Linear Model
print("Loading data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "immo_data.csv")
model_path = os.path.join(script_dir, "models", "linear_rent_model.pkl")

df = pd.read_csv(data_path)
linear_model = joblib.load(model_path)

# 2. Preparation (Added 'regio1' for Location)
# Standard cleaning
df = df[(df["livingSpace"] > 20) & (df["livingSpace"] < 250)]
df = df[(df["baseRent"] > 200) & (df["baseRent"] < 5000)]
df["rent_per_sqm"] = df["baseRent"] / df["livingSpace"]
df = df[(df["rent_per_sqm"] >= 7) & (df["rent_per_sqm"] <= 35)]
df = df[(df["yearConstructed"] > 1850) & (df["yearConstructed"] <= 2026)]

# FEATURE SELECTION: Added regio1 (Federal State)
numeric_features = ["livingSpace", "yearConstructed", "noRooms", "floor"]
categorical_features = ["condition", "interiorQual", "regio1"] # Added Location
binary_features = ["lift", "balcony", "newlyConst"]

features = numeric_features + categorical_features + binary_features
df = df[features + ["rent_per_sqm"]].dropna()

# 3. Create the "Fair Deal" Label
# Use the Linear model to predict price
reg_features = ["livingSpace", "yearConstructed", "noRooms", "floor", "condition", "interiorQual", "lift", "balcony", "newlyConst"]
predicted_log_price = linear_model.predict(df[reg_features])
predicted_price = np.expm1(predicted_log_price)

# TARGET DEFINITION: 1 = Fair (Price is <= Market Pred + 5% buffer), 0 = Expensive
# Adding a 5% buffer helps the model distinguish between 'noise' and 'actually expensive'
df["is_fair_deal"] = (df["rent_per_sqm"] <= (predicted_price * 1.05)).astype(int)

# 4. Train-Test Split
X = df[features]
y = df["is_fair_deal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Optimized Logistic Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        # Location encoding is vital here
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
        ("bin", "passthrough", binary_features)
    ]
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    # Degree 2 interactions help see things like (Munich * Small Apartment)
    ("interaction", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    # C=0.1 adds stronger regularization to prevent overfitting to noisy immo-data
    ("classifier", LogisticRegression(
        class_weight='balanced', 
        max_iter=2000, 
        C=0.1, 
        solver='lbfgs'
    ))
])

# 6. Train & Evaluate
print("Training Location-Aware Logistic Regression...")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n" + "="*40)
print("FINAL ENHANCED RESULTS")
print("="*40)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("="*40)

# 7. Save
joblib.dump(clf, os.path.join(script_dir, "models", "logistic_fairness_model.pkl"))

print("Generating Feature Importance plot...")

# 1. Extract feature names from the pipeline steps
# Step A: Get names after ColumnTransformer (Scalers + OneHot)
raw_names = clf.named_steps['preprocessor'].get_feature_names_out()

# Step B: Get names after Polynomial Interaction (e.g., 'livingSpace condition_well_kept')
poly_names = clf.named_steps['interaction'].get_feature_names_out(input_features=raw_names)

# 2. Get the coefficients from the Logistic Regression
coefficients = clf.named_steps['classifier'].coef_[0]

# 3. Create a DataFrame for easy sorting
importance_df = pd.DataFrame({
    'Feature': poly_names,
    'Weight': coefficients
})

# 4. Filter for Top 20 most impactful features (by absolute value)
importance_df['AbsWeight'] = importance_df['Weight'].abs()
top_features = importance_df.sort_values(by='AbsWeight', ascending=False).head(20)

# 5. Plotting
plt.figure(figsize=(12, 10))

# Create a color map: Green for 'Fair Deal' indicators, Red for 'Overpriced' indicators
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_features['Weight']]

sns.barplot(
    x='Weight', 
    y='Feature', 
    data=top_features, 
    palette=colors
)

plt.title('Top 20 Drivers of "Fair Deal" Probability', fontsize=16)
plt.xlabel('Coefficient Value (Positive = More Likely Fair, Negative = More Likely Overpriced)', fontsize=12)
plt.ylabel('Feature Interaction', fontsize=12)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Save the plot
plot_path = os.path.join(script_dir, "feature_importance.png")
plt.tight_layout()
plt.savefig(plot_path)
print(f"\nSUCCESS: Feature importance plot saved as '{plot_path}'")
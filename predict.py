import joblib
import pandas as pd
import numpy as np
import os

# 1. Load the Models
# This logic ensures the script finds your models folder automatically
script_dir = os.path.dirname(os.path.abspath(__file__))
reg_path = os.path.join(script_dir, "models", "linear_rent_model.pkl")
clf_path = os.path.join(script_dir, "models", "logistic_fairness_model.pkl")

# Verify that you have trained the models first
if not os.path.exists(reg_path) or not os.path.exists(clf_path):
    print(f"Error: Model files not found in {os.path.join(script_dir, 'models')}")
    print("Please run your training scripts (Linear and Logistic) before running this.")
    exit()

print("Loading models... Ready.")
reg_model = joblib.load(reg_path)
clf_model = joblib.load(clf_path)

def get_user_input():
    print("\n" + "="*50)
    print("      GERMAN RENT FAIRNESS AUDITOR (v2.0)      ")
    print("="*50)
    print("Enter the apartment details to evaluate:")
    
    try:
        data = {
            "livingSpace": [float(input("Living Space (m²): "))],
            "yearConstructed": [float(input("Year Constructed (e.g., 2015): "))],
            "noRooms": [float(input("Number of Rooms: "))],
            "floor": [float(input("Floor (0 for Ground): "))],
            "condition": [input("Condition (e.g., well_kept, mint_condition): ").strip().lower()],
            "interiorQual": [input("Interior Quality (e.g., normal, sophisticated): ").strip().lower()],
            "regio1": [input("Federal State (e.g., Berlin, Sachsen, Bayern): ").strip()],
            "lift": [int(input("Has Lift? (1 for Yes, 0 for No): "))],
            "balcony": [int(input("Has Balcony? (1 for Yes, 0 for No): "))],
            "newlyConst": [int(input("Is Newly Constructed? (1 for Yes, 0 for No): "))]
        }
        return pd.DataFrame(data)
    except ValueError:
        print("\nError: Please enter numbers for space, year, rooms, and 1/0 for binary fields.")
        return None

# 2. Prediction Engine
input_df = get_user_input()

if input_df is not None:
    try:
        # STEP A: Regression - What should the market price be?
        # Note: We drop 'regio1' for the linear model as it wasn't part of that training set
        reg_features = ["livingSpace", "yearConstructed", "noRooms", "floor", "condition", "interiorQual", "lift", "balcony", "newlyConst"]
        pred_log = reg_model.predict(input_df[reg_features])
        
        # Reverse the Log1p transformation to get real Euros
        pred_price_sqm = np.expm1(pred_log)[0]
        total_rent = pred_price_sqm * input_df['livingSpace'][0]
        
        # STEP B: Classification - Is this specific deal 'Fair' given the location?
        # This model includes 'regio1' (State) to adjust for regional bubbles
        fairness_proba = clf_model.predict_proba(input_df)[0][1]

        # 3. Final Result Report
        print("\n" + "*"*50)
        print("                AUDIT REPORT                  ")
        print("*"*50)
        print(f"Estimated Market Rent:   {pred_price_sqm:.2f} €/m²")
        print(f"Total Monthly Base Rent: {total_rent:.2f} €")
        print(f"Fair Deal Probability:   {fairness_proba:.2%}")
        print("-" * 50)
        
        if fairness_proba > 0.5:
            print("VERDICT: FAIR DEAL")
            print("Analysis: This price is competitive for this specific region and quality.")
        else:
            print("VERDICT: POTENTIALLY OVERPRICED")
            print("Analysis: This rent is significantly higher than market averages for this state.")
        print("*"*50 + "\n")

    except Exception as e:
        print(f"\nOops! An error occurred during prediction: {e}")
        print("Tip: Make sure the State name (e.g., 'Berlin') is spelled correctly.")
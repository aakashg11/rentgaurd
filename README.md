# RentGaurd: German Rent Fairness Auditor

RentGaurd is a machine learning tool designed to evaluate rental prices in Germany. It uses a dual-model approach to estimate market value and identify if a specific listing is a "Fair Deal" or "Overpriced."



## The Architecture
The project utilizes two distinct models working together:
1. **Ridge Regression:** Estimates the fair market price per m² based on apartment features.
2. **Logistic Regression:** A classification model with **Interaction Terms** that predicts the probability of a "Fair Deal" by accounting for regional price bubbles (e.g., Berlin vs. Sachsen).





## Tech Stack
* **Language:** Python 3.10+
* **Frameworks:** Scikit-Learn, Pandas, NumPy
* **Deployment:** CLI-based Inference Engine



## Performance & Insights
* **Location Awareness:** The model identifies that location (State) is the #1 driver of price fairness.
* **Feature Engineering:** Implemented interaction terms (e.g., `New Construction x Berlin`) to capture non-linear market trends.
* **Accuracy:** Successfully moved the baseline from a random guess to a statistically significant auditor.





## Project Structure
* `train_linear_model.py`: Script to train the price estimator.
* `train_logistic_model.py`: Script to train the fairness classifier.
* `predict.py`: Interactive tool for real-world auditing.
* `models/`: Contains the serialized `.pkl` model files.



## Usage
To audit an apartment, run the prediction script:
```bash
python3 predict.py

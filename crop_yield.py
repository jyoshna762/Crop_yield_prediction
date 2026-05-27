import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import gzip

# Load the dataset
df = pd.read_csv("yield_df.csv")

# Data Cleaning
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop_duplicates(inplace=True)

# Select relevant columns
col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[col]

# Define features and target variable
X = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

# Preprocessing
ohe = OneHotEncoder(drop='first')
scale = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('StandardScale', scale, [0, 1, 2, 3]),  # Features to scale
        ('OneHotEncode', ohe, [4, 5])            # Categorical features to encode
    ],
    remainder='passthrough'
)

# Fit and transform the training data
X_train_dummy = preprocessor.fit_transform(X_train)
X_test_dummy = preprocessor.transform(X_test)  # Only transform the test data

# Train models
models = {
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(random_state=0),
    "Gradient Boosting": GradientBoostingRegressor(random_state=0)
}

model_results = {}
y_preds = {}  # Store predictions for analysis
for name, model in models.items():
    model.fit(X_train_dummy, y_train)
    y_pred = model.predict(X_test_dummy)
    y_preds[name] = y_pred
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_results[name] = {"MAE": mae, "MSE": mse, "R2": r2}

# Select the best model based on MSE (lower is better)
best_model_name = min(model_results, key=lambda name: model_results[name]["MSE"])
best_model = models[best_model_name]

# Save the best model and preprocessor
joblib.dump(best_model, 'best_model.pkl', compress=0)

import gzip
import shutil

with open('best_model.pkl', 'rb') as f_in:
    with gzip.open('best_model.pkl.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

joblib.dump(preprocessor, 'preprocessor.pkl')

# Save predictions and metrics for app1.py
joblib.dump(y_test, "y_test.pkl")  # Save actual values
joblib.dump(y_preds[best_model_name], "y_pred.pkl")  # Save predictions of the best model
joblib.dump({"metrics": model_results, "best_model": best_model_name}, "model_results.pkl")

# Save metrics for display
metrics_df = pd.DataFrame(model_results).T
metrics_df.to_csv("model_metrics.csv", index=True)

print("Model training completed and artifacts saved successfully.")

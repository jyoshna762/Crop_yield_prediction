# crop_yield_prediction

Introduction: Crop yield prediction estimates the harvest quantity for a given area and time, transitioning from traditional methods (field surveys and expert knowledge) to modern machine learning (ML) techniques. ML enables analysis of vast datasets, including weather, soil, and farming practices, revealing complex patterns for accurate predictions.

Key Components:

Data Collection:


Weather: Temperature, rainfall, humidity.
Soil: Type, pH, nutrients.
Satellite Data: Vegetation indices, crop health.
Farming Practices: Irrigation, fertilizer use, crop rotation.
Preprocessing & Feature Engineering:


Clean data (remove duplicates, handle missing values).
Scale numerical features and encode categorical ones.
Select relevant variables influencing yield.
Machine Learning Models:

Regression Models: Decision Trees, Random Forests, Gradient Boosting.
Deep Learning: CNNs/RNNs for time-series and satellite imagery.
Performance is measured using metrics like MAE, MSE, and R².
Implementation Steps:

Data Cleaning: Load and prepare the dataset.
Feature Engineering: Scale numerical data, encode categorical variables.
Model Training: Train Decision Tree, Random Forest, and Gradient Boosting models.
Model Evaluation: Select the best-performing model based on MSE.
Model Saving: Save the best model and preprocessing pipeline.
Streamlit Application:

Prediction Section: Input data to predict crop yield.
Data Analysis Section: View model metrics and data distributions via box plots.
Results:

The Random Forest model achieved the best performance with the lowest MSE and highest R².
Visualizations (box plots) provide insights into feature distributions like rainfall, temperature, pesticides, and crop yield.
Predictions guide decisions, e.g., for Year = 2000, Rainfall = 800mm, Pesticides = 10T, Temp = 25°C, Area = 'India', Crop = 'Wheat':
Predicted yield = 45.78 hg/ha using Random Forest.
Conclusion: ML-driven crop yield prediction enhances productivity, mitigates risks, and promotes sustainable agriculture, marking a significant step towards food security. Combining traditional methods with technological advances, it empowers farmers for efficient and resilient agricultural practices.

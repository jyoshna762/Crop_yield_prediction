import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import gzip

# Load the best model and preprocessor from crop_yield.py
with gzip.open('best_model.pkl.gz', 'rb') as f_in:
    best_model = joblib.load(f_in)
preprocessor = joblib.load('preprocessor.pkl')

# Load the dataset for data analysis
df = pd.read_csv("yield_df.csv")

# Streamlit App styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.pexels.com/photos/23623461/pexels-photo-23623461/free-photo-of-green-field-on-the-hill.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1');  /* Path to your image */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }
    .input-label {
        font-size: 30px;  /* Increased font size */
        font-weight: bold; 
        color: white;
    }
    .input-container {
        background-color: rgba(0, 0, 0, 0.6); /* Background color for the input section */
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .prediction-result {
        background-color: white; 
        color: black; 
        padding: 30px; 
        border-radius: 10px; 
        font-size: 36px;  /* Larger font size */
        text-align: center; 
        margin-top: 30px;
        font-weight: bold;
    }
    .title {
        font-size: 50px;  /* Increased font size */
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .box-plot-description, .scatter-plot-description {
        background-color: rgba(56, 88, 35, 0.8); /* Dark background for descriptions */
        color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        font-size: 18px;
        height: 350px;  /* Increased height to match plot */
        overflow-y: auto;
    }
    .table-container {
        margin-top: 20px;
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add heading for "Crop Yield Prediction" (without background)
st.markdown(
    '<div class="title">Crop Yield Prediction 🌱</div>',
    unsafe_allow_html=True
)

# Sidebar Navigation
st.sidebar.title("Navigation")
view = st.sidebar.radio("Select View", ["Prediction", "Data Analysis"])

if view == "Prediction":
    st.markdown("<h3 style='text-align: center; '>Input Values</h3>", unsafe_allow_html=True)

    # Styled container for prediction inputs (without green block)
    with st.container():
        # Add label for "Year"
        st.markdown('<div class="input-label">Select a Year</div>', unsafe_allow_html=True)
        selected_date = st.date_input("", value=datetime(2000, 1, 1))
        Year = selected_date.year  # Extract year from selected date

        # Add label for "Average Rainfall"
        st.markdown('<div class="input-label">Average Rainfall (mm/year)</div>', unsafe_allow_html=True)
        average_rain_fall_mm_per_year = st.number_input("", value=0.0, key="rainfall")

        # Add label for "Pesticides"
        st.markdown('<div class="input-label">Pesticides (tonnes)</div>', unsafe_allow_html=True)
        pesticides_tonnes = st.number_input("", value=0.0, key="pesticides")

        # Add label for "Average Temperature"
        st.markdown('<div class="input-label">Average Temperature (°C)</div>', unsafe_allow_html=True)
        avg_temp = st.number_input("", value=0.0, key="temp")

        # Add label for "Area"
        st.markdown('<div class="input-label">Area (e.g., "Albania")</div>', unsafe_allow_html=True)
        Area = st.text_input("", key="area")

        # Add label for "Crop Item"
        st.markdown('<div class="input-label">Crop Item (e.g., "Maize")</div>', unsafe_allow_html=True)
        Item = st.text_input("", key="item")

        if st.button("Predict"):
            try:
                # Create feature array and transform
                features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
                transform_features = preprocessor.transform(features)
                predicted_yield = best_model.predict(transform_features)

                # Display prediction with custom styling
                st.markdown(
                    f'<div class="prediction-result">The predicted crop yield is: {predicted_yield[0]:.2f} hg/ha</div>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error in prediction: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

elif view == "Data Analysis":
    st.subheader("Exploratory Data Analysis")

    # Combine original data with new data
    combined_data = pd.concat([df], ignore_index=True)

    # Scatter Plots with Descriptions
    st.write("### Scatter Plots for Feature Relationships")
    scatter_cols = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'hg/ha_yield']
    scatter_descriptions = {
        'average_rain_fall_mm_per_year': "This scatter plot compares the rainfall values with crop yields. A positive relationship would indicate that higher rainfall contributes to better yields.",
        'pesticides_tonnes': "This scatter plot examines the relationship between pesticide usage and crop yield. It can help assess if more pesticides correlate with increased yields.",
        'avg_temp': "This scatter plot shows how average temperature affects crop yield. Extreme temperatures could negatively impact crop yields.",
        'hg/ha_yield': "This scatter plot shows how crop yield (in hg/ha) relates to itself over time or between regions, helping to visualize trends in yield."
    }

    for col in scatter_cols:
        st.write(f"#### Scatter Plot for {col}")
        
        col1, col2 = st.columns([1, 1])  # Split the layout into 2 columns
        with col1:
            fig, ax = plt.subplots(figsize=(12, 12))  # Match the boxplot size
            sns.scatterplot(data=combined_data, x=col, y='hg/ha_yield', ax=ax)
            st.pyplot(fig)
        with col2:
            st.markdown(
                f'<div class="scatter-plot-description">{scatter_descriptions[col]}</div>',
                unsafe_allow_html=True
            )

    # Box Plot Analysis with Descriptions
    st.write("### Feature Analysis with Box Plots")
    numeric_cols = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'hg/ha_yield']
    descriptions = {
        'average_rain_fall_mm_per_year': "This box plot shows the distribution of average annual rainfall. Outliers indicate unusual rainfall years.",
        'pesticides_tonnes': "This box plot highlights the distribution of pesticide usage in tonnes across the dataset.",
        'avg_temp': "This box plot illustrates the variation in average temperature across regions in the dataset.",
        'hg/ha_yield': "This box plot displays the yield in hectograms per hectare, showing its distribution and potential outliers."
    }

    for col in numeric_cols:
        st.write(f"#### Box Plot for {col}")

        col1, col2 = st.columns([1, 1])  # Split the layout into 2 columns
        with col1:
            fig, ax = plt.subplots(figsize=(12, 12))  # Increased height of the plot
            sns.boxplot(data=combined_data, x=col, ax=ax)
            st.pyplot(fig)
        with col2:
            st.markdown(
                f'<div class="box-plot-description">{descriptions[col]}</div>',
                unsafe_allow_html=True
            )

    # Predicted vs Actual Values (scatter plot)
    st.write("### Predicted vs Actual Values")
    # Assuming `y_test` and `y_pred` are available from the prediction model
    # Example: y_test = df['actual_yield'], y_pred = model.predict(input_data)
    y_test = df['hg/ha_yield']  # Replace this with actual test values
    y_pred = best_model.predict(preprocessor.transform(df.drop(columns=['hg/ha_yield'])))

    col1, col2 = st.columns([1, 1])  # Side-by-side layout
    with col1:
        fig1, ax1 = plt.subplots(figsize=(12, 12))  # Match boxplot size
        sns.scatterplot(x=y_test, y=y_pred, ax=ax1)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel("Actual Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title("Predicted vs Actual Values")
        st.pyplot(fig1)
    with col2:
        st.markdown(
            """
            <div class="box-plot-description">
            <strong>Description:</strong> 
            This scatter plot compares the predicted yield values against the actual yield values. 
            A perfect model would result in all points lying along the diagonal line.
            </div>
            """,
            unsafe_allow_html=True,
        )

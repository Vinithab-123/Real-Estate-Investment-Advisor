import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load Models ---
try:
    # Ensure this path is correct
    cls_pipeline = joblib.load('models/cls_pipeline.pkl')
    reg_pipeline = joblib.load('models/reg_pipeline.pkl')
except FileNotFoundError:
    st.error("Error: Model files not found. Ensure 'cls_pipeline.pkl' and 'reg_pipeline.pkl' are in the 'models/' directory.")
    st.stop()


st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

st.title("🏡 Real Estate Investment Advisor")
st.markdown("---")

# --- PLACEHOLDER DATA FOR VISUALS (REPLACE THIS) ---
# NOTE: Replace this with the actual top 5 features and scores extracted from your trained model!
TOP_5_FEATURES_DATA = pd.DataFrame({
    'Feature': ['Price_per_SqFt', 'City_Mumbai', 'Size_in_SqFt', 'Age_of_Property', 'BHK'],
    'Score': [0.45, 0.15, 0.10, 0.07, 0.05]
})
# --- END PLACEHOLDER ---


# --- 2. User Input Section ---
# ... (Keep your original input fields here, abbreviated for space) ...

# DUMMY_CITIES and DUMMY_STATES need to be defined here for the select boxes
DUMMY_CITIES = ['Chennai', 'Pune', 'Ludhiana', 'Jodhpur', 'Mumbai', 'Bangalore']
DUMMY_STATES = ['Tamil Nadu', 'Maharashtra', 'Punjab', 'Rajasthan', 'Karnataka']

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox('State', DUMMY_STATES)
    city = st.selectbox('City', DUMMY_CITIES)
    property_type = st.selectbox('Property Type', ['Apartment', 'Independent House'])
    bhk = st.slider('BHK', 1, 6, 3)
    size_sqft = st.number_input('Size in SqFt', min_value=500, max_value=10000, value=2500, step=100)
    year_built = st.number_input('Year Built', min_value=1950, max_value=2025, value=2010)

with col2:
    furnished_status = st.selectbox('Furnished Status', ['Furnished', 'Semi-furnished', 'Unfurnished'])
    transport = st.selectbox('Public Transport Accessibility', ['High', 'Medium', 'Low'])
    parking = st.selectbox('Parking Space', ['Yes', 'No'])
    security = st.selectbox('Security', ['Yes', 'No'])
    owner_type = st.selectbox('Owner Type', ['Owner', 'Builder', 'Broker'])

# Calculate Age_of_Property (used as a feature)
age_of_property = 2025 - year_built


# --- 3. Prediction Logic ---
if st.button("💰 Analyze Investment Potential"):
    
    # 3.1 Prepare Input DataFrame (using dummy Price_per_SqFt)
    input_data = pd.DataFrame([{
        'State': state, 'City': city, 'Property_Type': property_type, 'BHK': bhk,
        'Size_in_SqFt': size_sqft, 'Price_per_SqFt': size_sqft / 1000, # Dummy placeholder for price calculation
        'Year_Built': year_built, 'Furnished_Status': furnished_status,
        'Age_of_Property': age_of_property, 'Public_Transport_Accessibility': transport,
        'Parking_Space': parking, 'Security': security, 'Owner_Type': owner_type
    }])

    with st.spinner('Calculating investment potential...'):
        # 3.2 Classification Prediction (Good Investment)
        prediction_cls = cls_pipeline.predict(input_data)[0]
        prediction_proba = cls_pipeline.predict_proba(input_data)[0][1]

        # 3.3 Regression Prediction (Future Price)
        prediction_reg = reg_pipeline.predict(input_data)[0]

        # --- Display Results ---
        st.header("Investment Recommendation")
        
        # Display 1: Classification
        if prediction_cls == 1:
            st.success(f"✅ **GOOD INVESTMENT!** High potential for profit. (Confidence: {prediction_proba*100:.1f}%)")
            st.balloons()
        else:
            st.warning(f"⚠️ **MODERATE INVESTMENT.** Potential profit may be lower. (Confidence of being 'Good': {prediction_proba*100:.1f}%)")

        st.markdown("---")
        
        # Display 2: Regression
        st.header("Price Forecast (After 5 Years)")
        st.metric(label="Estimated Value After 5 Years", value=f"₹ {prediction_reg:.2f} Lakhs")


# --- 4. Final Insights Section (for Project Documentation/Presentation) ---

st.markdown("---")
st.header("✨ Project Analytics & Model Insights")

# 4.1 Feature Importance (Model Insight)
col_model, col_eda = st.columns(2)

with col_model:
    st.subheader("Top Drivers of Price (Model Feature Importance)")
    st.markdown("The model uses these features most heavily to determine the price forecast:")
    # Using Streamlit's built-in chart function for the Feature Importance
    st.bar_chart(TOP_5_FEATURES_DATA.set_index('Feature'))

with col_eda:
    st.subheader("Location Price Trend (EDA)")
    st.markdown("Average price trends by city from the dataset:")
    
    # 4.2 EDA Charts (Must ensure the image files are in the same directory)
    try:
        st.image('eda_price_trends_by_city.png', caption='Top 15 Cities by Average Price per SqFt')
    except FileNotFoundError:
        st.error("Missing EDA image: 'eda_price_trends_by_city.png'. Please run the EDA steps to generate the image.")

# Optional: Add the second EDA chart below the columns
st.markdown("---")
st.subheader("Infrastructure Impact (EDA)")
try:
    st.image('eda_accessibility_vs_investment.png', caption='Good Investment Rate by Public Transport Accessibility')
except FileNotFoundError:
    st.error("Missing EDA image: 'eda_accessibility_vs_investment.png'.")
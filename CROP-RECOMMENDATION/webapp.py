# ------------------------------
# SMART CROP RECOMMENDATION APP
# ------------------------------

# Suppress runtime warnings
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Core libraries
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sklearn

# ------------------------------
# Load and display banner image
# ------------------------------
if os.path.exists("crop.png"):
    img = Image.open("crop.png")
    st.image(img, use_container_width=True)

# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv('Crop_recommendation.csv')

# Features and labels
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------------
# Train or load RandomForest model
# ------------------------------
RF_MODEL_FILE = 'RF.pkl'

def train_model():
    """Train a RandomForest model and save it."""
    model = RandomForestClassifier(n_estimators=20, random_state=5)
    model.fit(X_train, y_train)
    # Store sklearn version for future checks
    model._sklearn_version = sklearn.__version__
    with open(RF_MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    return model

def load_model():
    """Load model safely; retrain silently if version mismatch or error."""
    if os.path.exists(RF_MODEL_FILE):
        try:
            with open(RF_MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
            if getattr(model, "_sklearn_version", None) != sklearn.__version__:
                # Silent retraining in background
                model = train_model()
        except Exception:
            model = train_model()
    else:
        model = train_model()
    return model

rf_model = load_model()

# Evaluate model (optional)
y_pred = rf_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

# ------------------------------
# Prediction function
# ------------------------------
def predict_crop(n, p, k, temp, hum, ph_val, rain):
    features = pd.DataFrame(
        [[n, p, k, temp, hum, ph_val, rain]],
        columns=['N','P','K','temperature','humidity','ph','rainfall']
    )
    return rf_model.predict(features)[0]

# ------------------------------
# Streamlit app interface
# ------------------------------
def main():
    st.markdown("<h1 style='text-align: center;'>SMART CROP RECOMMENDATION</h1>", unsafe_allow_html=True)
    
    st.sidebar.title("AgriSens")
    st.sidebar.header("Enter Crop Details")
    
    nitrogen = st.sidebar.number_input("Nitrogen (N)", 0.0, 140.0, 0.0, 0.1)
    phosphorus = st.sidebar.number_input("Phosphorus (P)", 0.0, 145.0, 0.0, 0.1)
    potassium = st.sidebar.number_input("Potassium (K)", 0.0, 205.0, 0.0, 0.1)
    temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 51.0, 0.0, 0.1)
    humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 0.0, 0.1)
    ph_val = st.sidebar.number_input("pH Level", 0.0, 14.0, 0.0, 0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 0.0, 0.1)

    st.sidebar.markdown(f"**Model Accuracy:** {accuracy*100:.2f}%")
    
    if st.sidebar.button("Predict"):
        inputs = np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph_val, rainfall])
        if np.isnan(inputs).any() or (inputs == 0).all():
            st.error("Please fill in all input fields with valid values before predicting.")
        else:
            crop = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph_val, rainfall)
            st.success(f"The recommended crop is: **{crop}**")
            
            image_path = os.path.join('crop_images', crop.lower() + '.jpg')
            if os.path.exists(image_path):
                st.image(image_path, caption=f"Recommended crop: {crop}", use_container_width=True)
            else:
                st.info("No image available for this crop.")

# ------------------------------
# Runagrisens web app the app
# ------------------------------
if __name__ == '__main__':
    main()

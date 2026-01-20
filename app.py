import streamlit as st
import pandas as pd
import joblib

# ===============================
# LOAD MODEL & METADATA
# ===============================
model = joblib.load("model.pkl")
dummy_columns = joblib.load("dummy_columns.pkl")
num_cols = joblib.load("num_cols.pkl")

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Insurance Claim Predictor", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🚗 Insurance Claim Prediction System</h1>", unsafe_allow_html=True)
st.write("Enter the vehicle and policy details below to predict the probability of an insurance claim.")

# ===============================
# USER INPUT
# ===============================
def user_input():
    data = {}

    with st.expander("🔢 Numeric Features", expanded=True):
        col1, col2, col3 = st.columns(3)
        data["policy_tenure"] = col1.number_input("Policy Tenure (years)", 0.0, 20.0, value=5.0)
        data["age_of_car"] = col1.number_input("Age of Car (years)", 0.0, 20.0, value=3.0)
        data["age_of_policyholder"] = col1.number_input("Age of Policyholder", 18.0, 100.0, value=30.0)
        data["population_density"] = col2.number_input("Population Density", 0.0, 10000.0, value=500.0)
        data["airbags"] = col2.slider("Airbags", 0, 6, value=2)
        data["displacement"] = col2.number_input("Displacement (cc)", 0.0, 5000.0, value=1500.0)
        data["cylinder"] = col3.number_input("Cylinder", 1, 12, value=4)
        data["gear_box"] = col3.slider("Gear Box", 4, 7, value=5)
        data["turning_radius"] = col3.number_input("Turning Radius (m)", 0.0, 10.0, value=5.0)
        data["Length"] = col1.number_input("Length (m)", 0.0, 10.0, value=4.0)
        data["width"] = col2.number_input("Width (m)", 0.0, 5.0, value=1.8)
        data["height"] = col3.number_input("Height (m)", 0.0, 5.0, value=1.5)
        data["Gross_weight"] = col1.number_input("Gross Weight (kg)", 0.0, 5000.0, value=1200.0)
        data["ncap_rating"] = col2.slider("NCAP Rating", 0, 5, value=4)

    with st.expander("🔤 Categorical Features", expanded=True):
        col1, col2 = st.columns(2)
        data["area_cluster"] = col1.selectbox("Area Cluster", ["C1","C2","C3","C4","C5","C6","C7","C8"], index=0)
        data["segment"] = col1.selectbox("Segment", ["A","B","C1","C2"], index=0)
        data["make"] = col2.text_input("Make (e.g. 1, 2, 3)", value="1")
        data["model"] = col2.text_input("Model (e.g. M1, M2)", value="M1")
        data["fuel_type"] = col1.selectbox("Fuel Type", ["Petrol","Diesel","CNG"], index=0)
        data["engine_type"] = col2.text_input("Engine Type", value="Petrol")
        data["rear_brakes_type"] = col1.selectbox("Rear Brakes Type", ["Drum","Disc"], index=1)
        data["transmission_type"] = col1.selectbox("Transmission Type", ["Manual","Automatic"], index=0)
        data["steering_type"] = col2.selectbox("Steering Type", ["Power","Manual"], index=0)
        data["max_power"] = col1.text_input("Max Power", value="100")
        data["max_torque"] = col2.text_input("Max Torque", value="150")

    with st.expander("✅ Safety & Feature Flags", expanded=True):
        binary_cols = [
            "is_esc","is_adjustable_steering","is_tpms","is_parking_sensors",
            "is_parking_camera","is_front_fog_lights","is_rear_window_wiper",
            "is_rear_window_washer","is_rear_window_defogger","is_brake_assist",
            "is_power_door_locks","is_central_locking","is_power_steering",
            "is_driver_seat_height_adjustable","is_day_night_rear_view_mirror",
            "is_ecw","is_speed_alert"
        ]
        for col in binary_cols:
            data[col] = st.selectbox(col.replace("_"," ").title(), ["Yes","No"], index=1)

    return pd.DataFrame([data])

input_df = user_input()

# ===============================
# DISPLAY INPUT
# ===============================
st.subheader("📋 User Input Data")
st.dataframe(input_df)

# ===============================
# PREPROCESSING
# ===============================
encoded_df = pd.get_dummies(input_df)

# Add all missing columns at once to avoid fragmentation
missing_cols = [col for col in dummy_columns if col not in encoded_df.columns]
if missing_cols:
    zeros_df = pd.DataFrame(0, index=encoded_df.index, columns=missing_cols)
    encoded_df = pd.concat([encoded_df, zeros_df], axis=1)

# Reorder columns to match training
encoded_df = encoded_df[dummy_columns]

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Predict Claim"):
    prob = model.predict_proba(encoded_df)[0][1]
    result = "CLAIM" if prob > 0.49 else "NO CLAIM"

    st.markdown("### 🏷️ Prediction Result")
    st.metric(label="Claim Status", value=result)
    st.progress(int(prob*100))
    st.info(f"Claim Probability: **{prob:.2%}**")

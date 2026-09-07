import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="Hypertension Prediction System",
    page_icon="❤️",
    layout="centered"
)

# Model location
MODEL_PATH = Path(__file__).parent / "hypertension_random_forest.pkl"


# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# Application title
st.title("Hypertension Prediction System")

st.write(
    "Enter the required health information below "
    "to predict hypertension status."
)

st.subheader("Patient Information")


# Input fields
age = st.number_input(
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=80.0,
    value=25.0,
    step=0.1
)

cholesterol = st.selectbox(
    "High Cholesterol",
    ["No", "Yes"]
)

diabetes = st.selectbox(
    "Diabetes",
    ["No", "Yes"]
)

smoking = st.selectbox(
    "Smoking History",
    ["No", "Yes"]
)

physical_activity = st.selectbox(
    "Physical Activity",
    ["No", "Yes"]
)


# Prediction button
if st.button(
    "Predict Hypertension Status",
    use_container_width=True
):

    # Convert inputs to model format
    input_data = pd.DataFrame([
        {
            "RIDAGEYR": age,
            "RIAGENDR": 1 if sex == "Male" else 0,
            "BMXBMI": bmi,
            "BPQ080": 1 if cholesterol == "Yes" else 0,
            "DIQ010": 1 if diabetes == "Yes" else 0,
            "SMQ020": 1 if smoking == "Yes" else 0,
            "Physical_Activity": 1 if physical_activity == "Yes" else 0
        }
    ])

    # Make prediction
    prediction = int(model.predict(input_data)[0])

    # Get confidence
    probability = model.predict_proba(input_data)[0]
    confidence = probability[prediction] * 100


    # Display result
    st.divider()
    st.subheader("Prediction Result")


    if prediction == 1:

        st.error("HYPERTENSION")

        st.write(
            "The model predicts hypertension status as positive."
        )

    else:

        st.success("NON-HYPERTENSION")

        st.write(
            "The model predicts hypertension status as negative."
        )


    # Display confidence
    st.metric(
        "Model Confidence",
        f"{confidence:.2f}%"
    )


    st.info(
        "This research system does not replace professional medical diagnosis.Please consult a qualified healthcare professional or see a doctor for proper medical  diagnosis.Please consult a qualified healthcare professional or see a doctor for proper medical evaluation and diagnosis."
    )


# Footer
st.divider()

st.caption(
    "Random Forest | NHANES 2017–2018 | 7 Predictor Variables"
)


import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------
MODEL_PATH = "tourism_project/deployment/best_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("✈️ Tourism Package Prediction")
st.write(
    "Enter customer details below to predict whether the customer "
    "is likely to purchase the tourism package."
)

# ---------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------
st.subheader("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18.0,
        max_value=100.0,
        value=35.0
    )

    type_of_contact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    duration_of_pitch = st.number_input(
        "Duration of Pitch",
        min_value=0.0,
        value=15.0
    )

    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    number_of_person_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        value=2
    )

    number_of_followups = st.number_input(
        "Number of Followups",
        min_value=0.0,
        value=3.0
    )

    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )

    preferred_property_star = st.number_input(
        "Preferred Property Star",
        min_value=1.0,
        max_value=5.0,
        value=3.0
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced", "Unmarried"]
    )

with col3:
    number_of_trips = st.number_input(
        "Number of Trips",
        min_value=0.0,
        value=3.0
    )

    passport = st.selectbox(
        "Passport",
        [0, 1]
    )

    pitch_satisfaction_score = st.selectbox(
        "Pitch Satisfaction Score",
        [1, 2, 3, 4, 5]
    )

    own_car = st.selectbox(
        "Own Car",
        [0, 1]
    )

    number_of_children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0.0,
        value=1.0
    )

    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=25000.0
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if st.button("Predict Package Purchase"):

    input_data = pd.DataFrame({
        "Age": [age],
        "TypeofContact": [type_of_contact],
        "CityTier": [city_tier],
        "DurationOfPitch": [duration_of_pitch],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfPersonVisiting": [number_of_person_visiting],
        "NumberOfFollowups": [number_of_followups],
        "ProductPitched": [product_pitched],
        "PreferredPropertyStar": [preferred_property_star],
        "MaritalStatus": [marital_status],
        "NumberOfTrips": [number_of_trips],
        "Passport": [passport],
        "PitchSatisfactionScore": [pitch_satisfaction_score],
        "OwnCar": [own_car],
        "NumberOfChildrenVisiting": [number_of_children_visiting],
        "Designation": [designation],
        "MonthlyIncome": [monthly_income]
    })

    try:
        model = load_model()

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(
                "Prediction: The customer is likely to purchase "
                "the tourism package."
            )
        else:
            st.warning(
                "Prediction: The customer is unlikely to purchase "
                "the tourism package."
            )

        st.metric(
            "Probability of Purchasing the Package",
            f"{probability:.2%}"
        )

        with st.expander("View Input Data"):
            st.dataframe(input_data)

    except FileNotFoundError:
        st.error(
            "Trained model not found. Please run the ML training "
            "pipeline first so best_model.joblib is created."
        )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

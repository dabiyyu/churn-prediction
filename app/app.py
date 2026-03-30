import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import os

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# Get absolute path to project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIGURES = os.path.join(ROOT, 'reports', 'figures')

# Load model and scaler
@st.cache_resource
def load_model():
    with open(os.path.join(ROOT, 'models', 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_scaler():
    with open(os.path.join(ROOT, 'models', 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    return scaler

@st.cache_resource
def load_feature_names():
    with open(os.path.join(ROOT, 'models', 'feature_names.pkl'), 'rb') as f:
        features = pickle.load(f)
    return features

model = load_model()
scaler = load_scaler()
feature_names = load_feature_names()

# Sidebar
st.sidebar.title("📡 Churn Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🔮 Predict", "📊 Insights", "🤖 Model Performance"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** IBM Telco Churn")
st.sidebar.markdown("**Model:** Gradient Boosting")
st.sidebar.markdown("**ROC-AUC:** 0.8501")

if page == "🔮 Predict":
    st.title("🔮 Customer Churn Risk Predictor")
    st.markdown("Enter customer details below to predict churn probability.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account Info")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type",
                                 ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])

    with col2:
        st.subheader("Services")
        internet_service = st.selectbox("Internet Service",
                                         ["Fiber optic", "DSL", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines",
                                       ["Yes", "No", "No phone service"])
        online_security = st.selectbox("Online Security",
                                        ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support",
                                     ["Yes", "No", "No internet service"])

    with col3:
        st.subheader("Demographics & Charges")
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        monthly_charges = st.number_input("Monthly Charges ($)",
                                           min_value=0.0, max_value=200.0,
                                           value=65.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)",
                                         min_value=0.0, max_value=10000.0,
                                         value=float(tenure * monthly_charges),
                                         step=10.0)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Churn Risk", use_container_width=True)

    if predict_btn:
        # Build input dict matching feature names
        input_dict = {f: 0 for f in feature_names}

        # Numerical features
        numerical_inputs = {
            'Tenure Months': tenure,
            'Monthly Charges': monthly_charges,
            'Total Charges': total_charges,
            'CLTV': 4527.0,
        }

        # Scale numerical features
        num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV']
        num_df = pd.DataFrame([{c: numerical_inputs[c] for c in num_cols}])
        scaled = scaler.transform(num_df)
        for i, col in enumerate(num_cols):
            if col in input_dict:
                input_dict[col] = scaled[0][i]

        # Binary features
        binary_map = {
            'Senior Citizen': 1 if senior_citizen == 'Yes' else 0,
            'Partner': 1 if partner == 'Yes' else 0,
            'Dependents': 1 if dependents == 'Yes' else 0,
            'Phone Service': 1 if phone_service == 'Yes' else 0,
            'Paperless Billing': 1 if paperless_billing == 'Yes' else 0,
        }
        for k, v in binary_map.items():
            if k in input_dict:
                input_dict[k] = v

        # One-hot encoded features
        onehot_map = {
            f'Multiple Lines_{multiple_lines}': 1,
            f'Internet Service_{internet_service}': 1,
            f'Online Security_{online_security}': 1,
            f'Tech Support_{tech_support}': 1,
            f'Contract_{contract}': 1,
            f'Payment Method_{payment_method}': 1,
        }
        for k, v in onehot_map.items():
            if k in input_dict:
                input_dict[k] = v

        # Predict
        input_df = pd.DataFrame([input_dict])
        prob = model.predict_proba(input_df)[0][1]

        # Display result
        st.markdown("---")
        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.metric("Churn Probability", f"{prob:.1%}")

        with col_r2:
            if prob >= 0.7:
                st.error("🔴 HIGH RISK")
            elif prob >= 0.4:
                st.warning("🟡 MEDIUM RISK")
            else:
                st.success("🟢 LOW RISK")

        with col_r3:
            st.metric("Retention Probability", f"{1-prob:.1%}")

        # Recommendation
        st.markdown("### 💡 Recommended Action")
        if prob >= 0.7:
            st.error("""
            **Immediate intervention required:**
            - Offer contract upgrade incentive (month-to-month → annual)
            - Escalate to senior customer service representative
            - Consider personalized retention discount
            """)
        elif prob >= 0.4:
            st.warning("""
            **Proactive monitoring recommended:**
            - Schedule a check-in call within 30 days
            - Offer loyalty reward or service upgrade
            - Review recent support ticket history
            """)
        else:
            st.success("""
            **Customer appears stable:**
            - Continue standard engagement
            - Consider upsell opportunities
            - Monitor at next quarterly review
            """)

elif page == "📊 Insights":
    st.title("📊 Key Business Insights")
    st.markdown("Findings from exploratory data analysis of 7,043 customers.")
    st.markdown("---")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Churn Rate", "27%")
    col2.metric("Customers Lost", "1,869")
    col3.metric("Avg Monthly Charge (Churned)", "$74.44")
    col4.metric("Total CLTV Lost", "$7.8M")

    st.markdown("---")

    # Charts
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Churn by Contract Type")
        st.image(os.path.join(FIGURES, '02_churn_by_contract.png'),
                 use_container_width=True)

    with col_b:
        st.subheader("Churn by Tenure Group")
        st.image(os.path.join(FIGURES, '03_churn_by_tenure.png'),
                 use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Monthly Charges Distribution")
        st.image(os.path.join(FIGURES, '04_monthly_charges_distribution.png'),
                 use_container_width=True)

    with col_d:
        st.subheader("Churn by Internet Service")
        st.image(os.path.join(FIGURES, '06_churn_by_internet.png'),
                 use_container_width=True)

    st.subheader("Top Churn Reasons")
    st.image(os.path.join(FIGURES, '05_churn_reasons.png'),
             use_container_width=True)
    
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", "Gradient Boosting")
    col2.metric("ROC-AUC", "0.8501")
    col3.metric("Churners Caught", "198 / 374")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("ROC Curve Comparison")
        st.image(os.path.join(FIGURES, '08_roc_curve_comparison.png'),
                 use_container_width=True)

    with col_b:
        st.subheader("Confusion Matrix")
        st.image(os.path.join(FIGURES, '09_confusion_matrix.png'),
                 use_container_width=True)

    st.subheader("Feature Importance (SHAP)")
    st.image(os.path.join(FIGURES, '11_shap_bar.png'),
             use_container_width=True)

    st.markdown("---")
    st.subheader("📌 Model Limitations & Future Improvements")
    st.markdown("""
    - **Recall is 53%** — half of churners are missed. Hyperparameter tuning and 
    threshold adjustment could improve this.
    - **Class imbalance** — SMOTE oversampling could help the model learn 
    churner patterns better.
    - **Feature engineering** — adding usage trend features (charge increase over time) 
    may improve predictions.
    - **Real-time scoring** — pipeline could be connected to a CRM system for 
    automated daily scoring.
    """)
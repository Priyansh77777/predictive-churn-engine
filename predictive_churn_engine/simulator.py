import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Intelligence System", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    div[data-testid="stMetricValue"] {font-size: 2rem;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    df = pd.read_csv('churn_data.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    X = pd.get_dummies(df.drop(columns=['Churn', 'customerID']), drop_first=True)
    y = df['Churn']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)
    
    return model, X.columns

model, expected_columns = load_model()

st.title("Customer Retention Intelligence")
st.markdown("Predictive Churn Scoring & CS Intervention Engine")

tab1, tab2 = st.tabs(["👤 Individual Profile Simulator", "📂 Batch Scoring Engine (CSV)"])

# --- TAB 1: INDIVIDUAL SIMULATOR ---
with tab1:
    with st.sidebar:
        st.markdown("### ⚙️ Customer Parameters")
        tenure = st.slider("Account Tenure (Months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 15.0, 120.0, 75.0, step=0.5)
        total_charges = tenure * monthly_charges
        
        st.markdown("### 🔌 Subscription Specs")
        contract = st.selectbox("Contract Tier", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Infrastructure", ["Fiber optic", "DSL", "No"])
        payment = st.selectbox("Billing Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        tech_support = st.selectbox("Premium Tech Support", ["No", "Yes", "No internet service"])

    input_data = {
        'tenure': tenure, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
        'InternetService_No': 1 if internet == "No" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment == "Electronic check" else 0,
    }

    input_df = pd.DataFrame(0, index=[0], columns=expected_columns)
    for key, value in input_data.items():
        if key in input_df.columns:
            input_df.at[0, key] = value

    churn_prob = model.predict_proba(input_df)[0][1] * 100
    
    if churn_prob < 40:
        risk_status, risk_color, gauge_color = "Stable", "normal", "#00cc96"
    elif churn_prob < 70:
        risk_status, risk_color, gauge_color = "At Risk", "off", "#FFA15A"
    else:
        risk_status, risk_color, gauge_color = "Critical Flight Risk", "inverse", "#EF553B"

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Predicted Churn Risk", f"{churn_prob:.1f}%", risk_status, delta_color=risk_color)
    kpi2.metric("Annual Revenue at Risk", f"${(monthly_charges * 12):,.2f}")
    kpi3.metric("Lifetime Value (Realized)", f"${total_charges:,.2f}")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 🎯 Real-Time Risk Gauge")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = churn_prob,
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(0, 204, 150, 0.1)"}, 
                    {'range': [40, 70], 'color': "rgba(255, 161, 90, 0.1)"}, 
                    {'range': [70, 100], 'color': "rgba(239, 85, 59, 0.1)"} 
                ]
            }
        ))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 🛠️ Automated PM Playbook")
        if churn_prob < 40:
            st.success("**Low Flight Risk:** Trigger upsell campaign for Two-Year Contract.")
        else:
            st.error(f"**Action Required ({risk_status}):**")
            if payment == "Electronic check":
                st.warning("- **Friction:** Send in-app prompt offering $10 credit to switch to Auto-Pay.")
            if internet == "Fiber optic":
                st.warning("- **Price/Value:** High-cost fiber users churning. Initiate proactive tech check-in.")
            if contract == "Month-to-month":
                st.warning("- **Vulnerability:** Offer 15% discount for migrating to a One-Year plan.")

# --- TAB 2: BATCH SCORING ---
with tab2:
    st.markdown("#### 📂 Upload Customer Cohort (CSV)")
    st.info("Upload a CSV to instantly score hundreds of users and generate a prioritized CS hit-list.")
    
    # --- Generate and Download Template ---
    template_cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                     'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
                     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                     'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                     'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
    template_df = pd.DataFrame([['CUST-0001', 'Female', 0, 'Yes', 'No', 12, 'Yes', 'No', 
                                 'Fiber optic', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 
                                 'Month-to-month', 'Yes', 'Electronic check', 75.0, 900.0]], 
                               columns=template_cols)
    csv_template = template_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download CSV Template",
        data=csv_template,
        file_name="churn_prediction_template.csv",
        mime="text/csv",
        help="Download a blank template with the exact column headers required by the ML model."
    )
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        if 'TotalCharges' in batch_df.columns:
            batch_df['TotalCharges'] = pd.to_numeric(batch_df['TotalCharges'], errors='coerce').fillna(0)
        
        score_df = batch_df.copy()
        if 'customerID' in score_df.columns:
            score_df = score_df.drop(columns=['customerID'])
        if 'Churn' in score_df.columns:
            score_df = score_df.drop(columns=['Churn'])
            
        X_batch = pd.get_dummies(score_df, drop_first=True)
        X_batch = X_batch.reindex(columns=expected_columns, fill_value=0)
        
        probs = model.predict_proba(X_batch)[:, 1] * 100
        batch_df['Churn Risk (%)'] = np.round(probs, 1)
        
        high_risk_users = batch_df[batch_df['Churn Risk (%)'] > 60].sort_values(by='Churn Risk (%)', ascending=False)
        revenue_at_risk = high_risk_users['MonthlyCharges'].sum() * 12 if 'MonthlyCharges' in high_risk_users.columns else 0
        
        st.error(f"⚠️ **{len(high_risk_users)} High-Risk Customers Identified** | 💸 **${revenue_at_risk:,.2f} Annual Revenue at Risk**")
        st.dataframe(high_risk_users[['customerID', 'Churn Risk (%)', 'MonthlyCharges', 'Contract', 'PaymentMethod'] if 'customerID' in high_risk_users.columns else high_risk_users.columns], use_container_width=True)

import shap
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from catboost import CatBoostClassifier
import joblib
import tempfile

Model_Path="/home/karuturividyasagar/platform/learning_rough/deeplearning/churn_prediction_project/cat_model_py3.pkl"
Data_Path="/home/karuturividyasagar/platform/learning_rough/deeplearning/churn_prediction_project/Telco-Customer-Churn.csv"
st.set_page_config(page_title="Churn prediction project")

@st.cache_resource

def load_data():
    data = pd.read_csv(Data_Path, encoding='ISO-8859-1', skiprows=[11])  # Skips line 12 (0-indexed) its making an issue
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')  # Convert TotalCharges to numeric, coerce errors to NaN
    data.dropna(subset=['TotalCharges'], inplace=True) 
    return data


def load_x_y(file_path):
    data=joblib.load(file_path)
    data.reset_index(drop=True, inplace=True)
    return data
    
def load_model():
    model = CatBoostClassifier()
    model = joblib.load(Model_Path)
    return model

def calculate_shap(model, X_train, X_test):
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_cat_train = explainer.shap_values(X_train)
    shap_values_cat_test = explainer.shap_values(X_test)
    return explainer, shap_values_cat_train, shap_values_cat_test

def plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, customer_id, X_test, X_train):
    # Visualize SHAP values for a specific customer
    customer_index = X_test[X_test['customerID'] == customer_id].index[0]
    fig, ax_2 = plt.subplots(figsize=(6,6), dpi=200)
    shap.decision_plot(explainer.expected_value, shap_values_cat_test[customer_index], X_test[X_test['customerID'] == customer_id], link="logit")
    st.pyplot(fig)
    plt.close()

def display_shap_summary(shap_values_cat_train, X_train):
    # Create the plot summarizing the SHAP values
    shap.summary_plot(shap_values_cat_train, X_train, plot_type="bar", plot_size=(12,12))
    summary_fig, _ = plt.gcf(), plt.gca()
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        summary_fig.savefig(tmpfile.name)
        st.image(tmpfile.name, use_column_width=True)
        plt.close()

def display_shap_waterfall_plot(explainer, expected_value, shap_values, feature_names, max_display=20):
    # Create SHAP waterfall drawing
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    shap.plots._waterfall.waterfall_legacy(expected_value, shap_values, feature_names=feature_names, max_display=max_display, show=False)
    st.pyplot(fig)
    plt.close()

def summary(model, data, X_train, X_test):
    # Calculate SHAP values
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)
    # Summarize and visualize SHAP values
    display_shap_summary(shap_values_cat_train, X_train)

def plot_shap(model, data, customer_id, X_train, X_test):
    # Calculate SHAP values
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)
    # Visualize SHAP values
    plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, customer_id, X_test, X_train)
    # Waterfall
    customer_index = X_test[X_test['customerID'] == customer_id].index[0]
    display_shap_waterfall_plot(explainer, explainer.expected_value, shap_values_cat_test[customer_index], feature_names=X_test.columns, max_display=20)

st.title("Telco customer churn prediction")

def main():
    model = load_model()
    data = load_data()

    X_train = load_x_y("X_train.pkl")
    X_test = load_x_y("X_test.pkl")
    y_train = load_x_y("y_train.pkl")
    y_test = load_x_y("y_test.pkl")

    max_tenure = data['tenure'].max()
    max_monthly_charges = data['MonthlyCharges'].max()
    max_total_charges = data['TotalCharges'].max()

    # Radio buttons for options
    election = st.radio("Make Your Choice:", ("Feature Importance", "User-based SHAP", "Calculate the probability of CHURN"))
    available_customer_ids = X_test['customerID'].tolist()
    
    # If User-based SHAP option is selected
    if election == "User-based SHAP":
        # Customer ID text input
        customer_id = st.selectbox("Choose the Customer", available_customer_ids)
        customer_index = X_test[X_test['customerID'] == customer_id].index[0]
        st.write(f'Customer {customer_id}: Actual value for the Customer Churn : {y_test.iloc[customer_index]}')
        y_pred = model.predict(X_test)
        st.write(f"Customer {customer_id}: CatBoost Model's prediction for the Customer Churn : {y_pred[customer_index]}")
        plot_shap(model, data, customer_id, X_train=X_train, X_test=X_test)
    
    # If Feature Importance is selected
    elif election == "Feature Importance":
        summary(model, data, X_train=X_train, X_test=X_test)

    # Calculate the probability of CHURN
    elif election == "Calculate the probability of CHURN":
        # Retrieving data from the user
        customerID = "6464-UIAEA"
        gender = st.selectbox("Gender:", ("Female", "Male"))
        senior_citizen = st.number_input("SeniorCitizen (0: No, 1: Yes)", min_value=0, max_value=1, step=1)
        partner = st.selectbox("Partner:", ("No", "Yes"))
        dependents = st.selectbox("Dependents:", ("No", "Yes"))
        tenure = st.number_input("Tenure:", min_value=0, max_value=max_tenure, step=1)
        phone_service = st.selectbox("PhoneService:", ("No", "Yes"))
        multiple_lines = st.selectbox("MultipleLines:", ("No", "Yes"))
        internet_service = st.selectbox("InternetService:", ("No", "DSL", "Fiber optic"))
        online_security = st.selectbox("OnlineSecurity:", ("No", "Yes"))
        online_backup = st.selectbox("OnlineBackup:", ("No", "Yes"))
        device_protection = st.selectbox("DeviceProtection:", ("No", "Yes"))
        tech_support = st.selectbox("TechSupport:", ("No", "Yes"))
        streaming_tv = st.selectbox("StreamingTV:", ("No", "Yes"))
        streaming_movies = st.selectbox("StreamingMovies:", ("No", "Yes"))
        contract = st.selectbox("Contract:", ("Month-to-month", "One year", "Two year"))
        paperless_billing = st.selectbox("PaperlessBilling", ("No", "Yes"))
        payment_method = st.selectbox("PaymentMethod:", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=max_monthly_charges, step=0.01)
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=max_total_charges, step=0.01)
        
        # Confirmation button
        confirmation_button = st.button("Confirm")

        # When the confirmation button is clicked
        if confirmation_button:
            # Converting user-entered data into a data frame
            new_customer_data = pd.DataFrame({
                "customerID": [customerID],
                "gender": [gender],
                "SeniorCitizen": [senior_citizen],
                "Partner": [partner],
                "Dependents": [dependents],
                "tenure": [tenure],
                "PhoneService": [phone_service],
                "MultipleLines": [multiple_lines],
                "InternetService": [internet_service],
                "OnlineSecurity": [online_security],
                "OnlineBackup": [online_backup],
                "DeviceProtection": [device_protection],
                "TechSupport": [tech_support],
                "StreamingTV": [streaming_tv],
                "StreamingMovies": [streaming_movies],
                "Contract": [contract],
                "PaperlessBilling": [paperless_billing],
                "PaymentMethod": [payment_method],
                "MonthlyCharges": [monthly_charges],
                "TotalCharges": [total_charges]
            })

            # Predict churn probability using the model
            churn_probability = model.predict_proba(new_customer_data)[:, 1]

            # Format churn probability
            formatted_churn_probability = "{:.2%}".format(churn_probability.item())

            # Display user-entered data
            st.subheader("User-entered Data:")
            st.write("Customer ID:", customerID)
            st.write("Gender:", gender)
            st.write("Senior Citizen:", senior_citizen)
            st.write("Partner:", partner)
            st.write("Dependents:", dependents)
            st.write("Tenure:", tenure)
            st.write("Phone Service:", phone_service)
            st.write("Multiple Lines:", multiple_lines)
            st.write("Internet Service:", internet_service)
            st.write("Online Security:", online_security)
            st.write("Online Backup:", online_backup)
            st.write("Device Protection:", device_protection)
            st.write("Tech Support:", tech_support)
            st.write("Streaming TV:", streaming_tv)
            st.write("Streaming Movies:", streaming_movies)
            st.write("Contract:", contract)
            st.write("Paperless Billing:", paperless_billing)
            st.write("Payment Method:", payment_method)
            st.write("Monthly Charges:", monthly_charges)
            st.write("Total Charges:", total_charges)
            # Display churn probability
            st.subheader("Churn Probability:")
            st.write("Probability of Churn:", formatted_churn_probability)



if __name__ == "__main__":
    main()

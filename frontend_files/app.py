import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")
st.title("SuperKart Sales Revenue Predictor")
st.write("Enter product and store details below to forecast quarterly sales revenue.")

# Backend URL (uses Docker container hostname 'backend' on the shared Docker network)
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860")

st.subheader("Single Product Prediction")

col1, col2 = st.columns(2)
with col1:
    product_weight = st.number_input("Product Weight", min_value=0.0, value=12.66, step=0.1)
    product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    product_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.027, step=0.005, format="%.3f")
    product_mrp = st.number_input("Product MRP", min_value=0.0, value=117.08, step=1.0)
    product_id_char = st.selectbox("Product ID Prefix", ["FD", "DR", "NC"])

with col2:
    store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    store_city_tier = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
    store_age = st.number_input("Store Age (Years)", min_value=0, value=16, step=1)
    product_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

if st.button("Predict Sales"):
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar,
        "Product_Allocated_Area": product_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_city_tier,
        "Store_Type": store_type,
        "Product_Id_char": product_id_char,
        "Store_Age_Years": store_age,
        "Product_Type_Category": product_category,
    }
    try:
        res = requests.post(f"{BACKEND_URL}/v1/predict", json=payload)
        if res.status_code == 200:
            predicted_sales = res.json().get("Sales")
            st.success(f"Predicted Product Store Sales: ${predicted_sales:,.2f}")
        else:
            st.error(f"API Error ({res.status_code}): {res.text}")
    except Exception as e:
        st.error(f"Connection Error: {e}")

st.divider()
st.subheader("Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload Batch CSV File", type=["csv"])

if uploaded_file is not None and st.button("Predict Batch Sales"):
    try:
        files = {"file": uploaded_file.getvalue()}
        res = requests.post(f"{BACKEND_URL}/v1/predictbatch", files=files)
        if res.status_code == 200:
            preds = res.json()
            batch_df = pd.read_csv(uploaded_file)
            batch_df["Predicted_Sales"] = [preds[str(i)] for i in range(len(batch_df))]
            st.dataframe(batch_df)
        else:
            st.error(f"API Error ({res.status_code}): {res.text}")
    except Exception as e:
        st.error(f"Connection Error: {e}")

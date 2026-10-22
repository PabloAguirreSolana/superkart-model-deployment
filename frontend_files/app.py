import os
import requests
import streamlit as st

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="🛒",
    layout="centered"
)

API_URL = os.getenv(
    "API_URL",
    "http://backend:5000/predict"
)

st.title("SuperKart Sales Prediction")
st.write("Predict product-level sales revenue using the trained ML model.")

with st.form("prediction_form"):

    st.subheader("Product Information")

    product_weight = st.number_input(
        "Product Weight",
        min_value=0.01,
        value=12.65
    )

    sugar_content = st.selectbox(
        "Sugar Content",
        ["Low Sugar", "Regular", "No Sugar"]
    )

    allocated_area = st.number_input(
        "Product Allocated Area",
        min_value=0.001,
        max_value=1.0,
        value=0.069,
        format="%.3f"
    )

    product_mrp = st.number_input(
        "Product MRP",
        min_value=0.01,
        value=147.03
    )

    product_prefix = st.selectbox(
        "Product ID Prefix",
        ["FD", "NC", "DR"]
    )

    product_category = st.selectbox(
        "Product Type Category",
        ["Perishables", "Non Perishables"]
    )

    st.subheader("Store Information")

    store_size = st.selectbox(
        "Store Size",
        ["Small", "Medium", "High"]
    )

    city_type = st.selectbox(
        "City Tier",
        ["Tier 1", "Tier 2", "Tier 3"]
    )

    store_type = st.selectbox(
        "Store Type",
        [
            "Departmental Store",
            "Supermarket Type1",
            "Supermarket Type2",
            "Food Mart"
        ]
    )

    store_age = st.number_input(
        "Store Age in Years (as of 2025)",
        min_value=0,
        value=16,
        step=1
    )

    submitted = st.form_submit_button(
        "Predict Sales",
        use_container_width=True
    )

if submitted:

    input_data = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar_content,
        "Product_Allocated_Area": allocated_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type,
        "Product_Id_char": product_prefix,
        "Store_Age_Years": store_age,
        "Product_Type_Category": product_category
    }

    try:
        response = requests.post(
            API_URL,
            json=input_data,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()
        prediction = result["predictions"][0]

        st.success("Prediction completed successfully!")

        st.metric(
            "Predicted Sales Revenue",
            f"{prediction:,.2f}"
        )

    except requests.exceptions.RequestException as e:
        st.error(f"API connection error: {e}")

    except (KeyError, IndexError, TypeError, ValueError):
        st.error("Unexpected response from the prediction API.")

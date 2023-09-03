import streamlit as st
import os
from st_helper_func import remove_top_space_canvas, gridbuilder_config_setup, load_data, navbar_edit

# Page config
st.set_page_config(layout = "wide")

# Remove empty space on top. Use rem for root element font-size referencing (18px)
remove_top_space_canvas()
navbar_edit()

#########################
# Dataset interaction
#########################
st.write("## Datasets information")

# os.getcwd() takes into account of the folder path which streamlit setup is run. In this case it is under time_series parent folder
data_dir = os.path.join(os.getcwd(), 'data')

# Path to data files
fulfilment_center_info = os.path.join(data_dir , 'fulfilment_center_info.csv')
meal_info = os.path.join(data_dir , 'meal_info.csv')
train_info = os.path.join(data_dir , 'train.csv')
test_info = os.path.join(data_dir , 'test.csv')

# Read dataframe using function call with st.cache_data decorator
fulfilment_df = load_data(fulfilment_center_info, parse_date=False)
meal_df = load_data(meal_info, parse_date=False)
train_df = load_data(train_info, parse_date=False)
test_df = load_data(test_info, parse_date=False)

with st.container():
    col1, col2 = st.columns([1,3])
    with col1:
        st.subheader("Data dictionary for fulfilment center")
        st.markdown("""
            |Variable|Definition|
            |---|---|
            |center_id|Unique ID for fulfillment center|
            |city_code|Unique code for city|
            |region_code|Unique code for region|
            |center_type|Anonymized center type|
            |op_area|Area of operation (in km^2)|
            """)
    with col2:
        st.subheader("Fulfilment center dataset")
        ag_fulfilment_df = gridbuilder_config_setup(fulfilment_df)

st.markdown("")
st.markdown("")
with st.container():
    col1, col2 = st.columns([1,3])
    with col1:
        st.subheader("Data dictionary for meal information")
        st.markdown("""
            |Variable|Definition|
            |---|---|
            |meal_id|Unique ID for the meal|
            |category|Type of meal (beverages/snacks/soups….)|
            |cuisine|Meal cuisine (Indian/Italian/…)|
            """)

    with col2:
        st.subheader("Meal information dataset")
        ag_meal_df = gridbuilder_config_setup(meal_df)

st.markdown("")
st.markdown("")
with st.container():
    col1, col2 = st.columns([1,3])
    with col1:
        st.subheader("Data dictionary for weekly demand (training dataset)")
        st.markdown("""
            |Variable|Definition|
            |---|---|
            |id|Unique ID|
            |week|Week No|
            |center_id|Unique ID for fulfillment center|
            |meal_id|Unique ID for Meal|
            |checkout_price|Final price including discount, taxes & delivery charges|
            |base_price|Base price of the meal|
            |emailer_for_promotion|Emailer sent for promotion of meal|
            |homepage_featured|Meal featured at homepage|
            |num_orders|(Target) Orders Count|
        """)
    with col2:
        st.subheader("Weekly demand dataset (training dataset)")
        ag_train_df = gridbuilder_config_setup(train_df)

st.markdown("")
st.markdown("")
with st.container():
    col1, col2 = st.columns([1,3])
    with col1:
        st.subheader("Data dictionary for weekly demand (test dataset) without any order information")
        st.markdown("""
            |Variable|Definition|
            |---|---|
            |id|Unique ID|
            |week|Week No|
            |center_id|Unique ID for fulfillment center|
            |meal_id|Unique ID for Meal|
            |checkout_price|Final price including discount, taxes & delivery charges|
            |base_price|Base price of the meal|
            |emailer_for_promotion|Emailer sent for promotion of meal|
            |homepage_featured|Meal featured at homepage|
        """)
    with col2:
        st.subheader("Weekly demand dataset (testing dataset)")
        ag_test_df = gridbuilder_config_setup(test_df)
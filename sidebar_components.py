import streamlit as st
import numpy as np
import pandas as pd
import os

def load_sidebar(df):
    """This function defines and constructs the general structure of the sidebar of streamlit web application and provides suitable option selection as extracted from input dataframe.
    
    Args:
        df (dataframe): Input dataframe

    Returns:
        start_date, end_date, call_center_option, region_code_option, city_code_option, cuisine_cat_option, weeks_to_predict_option,model_type_option, model_selection_option (streamlit selection widgets):
            Widgets that offers user interaction with streamlit application.
            
    Raises:
        None

    """
    
    ### Filters for visualisation
    # Bold markdown

    min_date = df['date'].min()
    max_date = df['date'].max()

    # Use date input interface with only valid dates to choose
    st.sidebar.markdown("**Start date for visualisation**")
    start_date = st.sidebar.date_input('', value=min_date, min_value= min_date, max_value= max_date)

    st.sidebar.markdown("**End date for visualisation**")
    end_date = st.sidebar.date_input('', value=max_date, min_value= min_date, max_value= max_date)

    if start_date <= end_date:
        pass
    else:
        st.error('Start date must be > End date')

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    """ # Call Center
    call_center_id = set(df['center_id_type'].unique())
    call_center_option = st.sidebar.selectbox(
        'Call Center ID-Center Type',
        call_center_id)

    # Region code
    region_code = set(df['region_code'].unique())
    region_code_option = st.sidebar.selectbox(
        'Region of interest',
        region_code)

    # City code is related to region as city is a smaller geographic area residing in a region
    city_code = set(df[df['region_code']==region_code_option]['city_code'].unique())
    city_code_option = st.sidebar.selectbox(
        'City of interest',
        city_code)

    # Cuisine-category type
    cuisine_cat_types = set(df['meal_id-cuisine-category'].unique())
    cuisine_cat_option = st.sidebar.selectbox(
        'Meal ID-Cuisine-Category option',
        cuisine_cat_types) """


    st.sidebar.markdown("")
    st.sidebar.markdown("")
 

    ### Filters for modelling
    # Bold markdown
    st.sidebar.markdown("**Select weeks to forecast**")
    # Future weeks to predict
    weeks_to_predict = set(np.arange(2, 11))
    weeks_to_predict_option = st.sidebar.selectbox(
        '',
        weeks_to_predict)

    st.sidebar.markdown("**Select model type**")
    # Models to use for forecasting
    model_types = ('Statistical Model', 'Machine Learning', 'Deep Learning')

    model_type_option = st.sidebar.selectbox(
        '',
        model_types)
    

    model_dir = os.path.join('models', model_type_option)
    pkl_file_list = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]

    if len(pkl_file_list) > 0:
        # Widget to select model file
        st.sidebar.markdown("**Select Model file**")
        model_selection = set(pkl_file_list)
        model_selection_option = st.sidebar.selectbox(
            '',
            model_selection)
    else:
        st.sidebar.markdown("")
        model_selection_option = 'None'

    #return start_date, end_date, call_center_option, region_code_option, city_code_option, cuisine_cat_option, weeks_to_predict_option,model_type_option, model_selection_option

    return start_date, end_date, weeks_to_predict_option,model_type_option, model_selection_option
"""
Dashboard design
"""

import streamlit as st # data web application development
import os 
import pandas as pd
import plotly.express as px # interactive charts

from src.sarima_model import SarimaModel, split_data
from general_helper_func import extract_info_from_model_name,resample_num_orders_weekly, simple_feature_engineer, create_test_data_date_info
from sidebar_components import load_sidebar
from st_helper_func import remove_top_space_canvas, navbar_edit, load_data, forecast_default_template_for_no_model, add_text_to_filter_pane, adjust_filter_font
from dashboard_visuals_func import plot_order_processed_by_center, plot_cuisine_preference, plot_weekly_order_trend

# Set page config:
st.set_page_config( 
    layout = "wide", # Set to wide
)
remove_top_space_canvas()
navbar_edit()
add_text_to_filter_pane()
adjust_filter_font()

# Data import. Note that training data has been preprocessed and saved as merged.csv, unlike test data which is untouched
save_merged_df = os.path.join(os.getcwd(), 'data', 'merged.csv')
test_file_path = os.path.join(os.getcwd(), 'data', 'test.csv')

df = load_data(save_merged_df, parse_date=True)

# Do for test data
df_test = load_data(test_file_path, parse_date=False)
df_test = create_test_data_date_info(df_test)


# For filter widget dropdown selection
n_cities = df['city_code'].nunique()
n_regions = df['region_code'].nunique()
n_cuisines = df['cuisine'].nunique()

# Combine features
df = simple_feature_engineer(df)

#####################
# Canvas structure
#####################

######################
### Sidebar
#####################

start_date, end_date, weeks_to_predict_option,model_type_option, model_selection_option = load_sidebar(df)

#####################
# Canvas Intro/Header
#####################

st.title(f'Meal Operations Interactive Dashboard - Data summary for selected time period: {start_date.date()} to {end_date.date()} ')

################################
# Numeric display of data
################################
info_fig_col1, info_fig_col2, info_fig_col3, info_fig_col4, info_fig_col5, info_fig_col6, info_fig_col7= st.columns(7, gap = 'large')

filtered_time_df = df[(df['date']<= end_date ) & (df['date']>= start_date)]
total_order = filtered_time_df['num_orders'].sum()
info_fig_col1.metric('Total orders', f'{total_order:,}')

# Calculate total checkout price and convert to Millions for visuals
sum_base_price = filtered_time_df['total_base_price'].sum()

if sum_base_price >=1e6:
    rnd_sum_base_price = sum_base_price // 1e6
    info_fig_col2.metric('Total base price ', f'{rnd_sum_base_price:,}Mil')
else:
    rnd_sum_base_price = sum_base_price
    info_fig_col2.metric('Total base price', f'{rnd_sum_base_price:,}')

# Calculate total checkout price and convert to Millions for visuals
sum_checkout_price = filtered_time_df['total_checkout_price'].sum()

if sum_checkout_price >=1e6:
    rnd_sum_checkout_price = sum_checkout_price // 1e6
    info_fig_col3.metric('Total order checkout price', f'{rnd_sum_checkout_price:,}Mil')
else:
    rnd_sum_checkout_price = sum_checkout_price 
    info_fig_col3.metric('Total order checkout price', f'{rnd_sum_checkout_price:,}')


highest_demand_reg = filtered_time_df.groupby('region_code')['num_orders'].sum().idxmax()
info_fig_col4.metric('Highest demand region', highest_demand_reg)

lowest_demand_reg = filtered_time_df.groupby('region_code')['num_orders'].sum().idxmin()
info_fig_col5.metric('Lowest demand region', lowest_demand_reg)

most_sought_cuisine = filtered_time_df.groupby('cuisine')['num_orders'].sum().idxmax()
info_fig_col6.metric('Most sought cuisine', most_sought_cuisine)

least_sought_cuisine = filtered_time_df.groupby('cuisine')['num_orders'].sum().idxmin()
info_fig_col7.metric('Least sought cuisine', least_sought_cuisine)

################################
# Graphical plots of data
################################
desc_fig_col1, desc_fig_col2, desc_fig_col3 = st.columns(3, gap='large')

with desc_fig_col1:
    plot_weekly_order_trend(filtered_time_df)

# To change to show demand for specific cuisine instead of meal price
with desc_fig_col2:
    plot_cuisine_preference(filtered_time_df)

with desc_fig_col3:
    plot_order_processed_by_center(filtered_time_df)
    
###############################
# Forecast/Inference section
###############################
test_min_date = df_test.index.min().date()
test_max_date = (df_test.index.min() + pd.to_timedelta(weeks_to_predict_option, unit='W')).date()


if weeks_to_predict_option==1:
    st.title(f'Forecasted meal demand for next {weeks_to_predict_option} week ({test_min_date} to {test_max_date})')
else:
    st.title(f'Forecasted meal demand for next {weeks_to_predict_option} weeks ({test_min_date} to {test_max_date})')

# Uses model_name_option , weeks_to_predict_option
forecast_col1, forecast_col2 = st.columns([3,1])
with forecast_col1:
    if model_type_option == 'Statistical Model':
        model_name = str(model_selection_option.split('_')[0]).upper()
        st.subheader(f'Using {model_name} model')

        # Resample by weeks using num_orders columns as we are not predicting order by center id
        train = resample_num_orders_weekly(df)

        # Since we need to predict next time periods, create a dummy dataframe from test data sampled weekly
        date = pd.date_range(test_min_date,
                             periods=weeks_to_predict_option,
                             freq='W')
        test = pd.Series(data=None, index=date)
        #test = df_test['center_id'].resample('W').mean()

    
        model_path_to_load = os.path.join("models", model_type_option, model_selection_option)
 
        order_info, seasonal_order_info = extract_info_from_model_name(model_selection_option)
        
        # Instantiate model class and use its load function
        model = SarimaModel(ts = train, test = test, order=order_info, seasonal_order=seasonal_order_info)
        #model = ArimaModel(weeks_to_predict=weeks_to_predict_option, ts = train)
        
        # Get train/test split that is conducted during class instantiation
        #train_df = train.to_frame()
        test_df = test.to_frame()
 
        # Load model
        model.load_model(load_model_path = model_path_to_load)
        forecast_col_name = 'Forecasted units'
        test_df[forecast_col_name] = model.forecast(n_periods=weeks_to_predict_option)
        col_to_drop = [col for col in test_df.columns if col!=forecast_col_name]
        test_df = test_df.drop(col_to_drop, axis = 1)

        # Show plotly charts
        fig = px.line(test_df, markers=True)
        fig.update_layout(yaxis_title=None, xaxis_title=None, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        #st.write('Forecast not available at the moment')

    # Placeholder with no forecast
    elif model_type_option  == 'Machine Learning':
        forecast_default_template_for_no_model()

    elif model_type_option  == 'Deep Learning':
        forecast_default_template_for_no_model()
    else:
        raise NotImplementedError

with forecast_col2:
    st.subheader('Forecasted meal demands at a glance')
    st.table(test_df)

    st.subheader('Model performance using last 10 weeks data')
    # To create train test sets from existing training time series data
    train, test = split_data(train)

    # Update with sample train and test data for evaluation
    model.train = train
    model.test = test
    forecast = model.forecast(n_periods = test.shape[0])

    # Evaluate
    eval_metric = model.evaluate(test, forecast)
    st.metric('RMSE', f'{eval_metric:.2f}')

"""
Helper functions for dashboard building
"""
import streamlit as st
import plotly.express as px

def plot_weekly_order_trend(df):
    """Plot time series data on total orders filtered for a given period 
    """
    fig_col1_title = "<h1 style='text-align: center; color:Grey'> Total weekly orders</h1>"
    st.markdown(fig_col1_title, unsafe_allow_html=True)
    weekly_order_df = df.groupby('date')['num_orders'].sum()

    tab1, tab2 = st.tabs(["Original", "Rolling mean(4 weeks)"])

    with tab1:
        weekly_order_filtered = weekly_order_df.to_frame()
        fig = px.line(weekly_order_filtered, markers=True)
        fig.update_layout(yaxis_title=None, xaxis_title=None, showlegend=False)
        st.plotly_chart(fig)
    with tab2:
        fig = px.line(weekly_order_filtered['num_orders'].rolling(4).mean(), markers=True)
        fig.update_layout(yaxis_title=None, xaxis_title=None, showlegend=False)
        st.plotly_chart(fig)

def plot_selected_meal_prices(df, cuisine_cat_option):
    """Plot time series data of selected meal price.
    """
    fig_col2_title = f"<h1 style='text-align: center; color:Grey'> '{cuisine_cat_option}' meal price</h1>"
    st.markdown(fig_col2_title, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Original", "Rolling mean(4 weeks)"])
    cuisine_cat_df = df[df['meal_id-cuisine-category']==cuisine_cat_option]
    cuisine_order = cuisine_cat_df.groupby('date')[['base_price','checkout_price']].mean()

    with tab1:
        fig = px.line(cuisine_order, markers=True)
        fig.update_layout(yaxis_title=None, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True, height=200)

    with tab2:
        fig = px.line(cuisine_order[['base_price','checkout_price']].rolling(4).mean(), markers=True)
        fig.update_layout(yaxis_title=None, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True, height=200)


def plot_demands_handled_by_center(df, call_center_option):
    fig_col3_title = f"<h1 style='text-align: center; color:Grey'>Orders processed by center: {call_center_option}</h1>"
    st.markdown(fig_col3_title, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Original", "Rolling mean(4 weeks)"])

    call_center_df = df[df['center_id_type']==call_center_option]
    call_center_order = call_center_df.groupby('date')['num_orders'].sum()
    with tab1:
        fig = px.line(call_center_order, markers=True)
        fig.update_layout(yaxis_title=None, xaxis_title=None, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, height=200)

    with tab2:
        call_center_order_df = call_center_order.to_frame()
        fig = px.line(call_center_order_df['num_orders'].rolling(4).mean(), markers=True)
        fig.update_layout(yaxis_title=None, xaxis_title=None, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, height=200)

def plot_order_processed_by_center(df):
    #plot_demands_handled_by_center(filtered_time_df, call_center_option)
    fig_col3_title = "<h1 style='text-align: center; color:Grey'> Orders processed by center type</h1>"
    st.markdown(fig_col3_title, unsafe_allow_html=True)

    center_type_order_df = df.groupby(['date','center_type'])['num_orders'].sum()

    center_type_order_mean_df= df.groupby(['date','center_type'])\
        ['num_orders'].mean()
    
    center_type_order_df = center_type_order_df.reset_index()
    center_type_order_mean_df = center_type_order_mean_df.reset_index()

    tab1, tab2 = st.tabs(["Original", "Mean orders per center by type"])
    with tab1:
        fig = px.line(center_type_order_df,
                    x="date",
                    color="center_type",
                    y='num_orders',
                    markers=True
        )
        fig.update_layout(yaxis_title=None, xaxis_title=None, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = px.line(center_type_order_mean_df,
                    x="date",
                    color="center_type",
                    y='num_orders',
                    markers=True
        )
        fig.update_layout(yaxis_title=None, xaxis_title=None, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)



def plot_cuisine_preference(df):
    fig_col2_title = "<h1 style='text-align: center; color:Grey'> Cuisine preference over the period</h1>"
    st.markdown(fig_col2_title, unsafe_allow_html=True)
    cuisine_order_df = df.groupby('cuisine')['num_orders'].sum()

    cuisine_order = cuisine_order_df.to_frame()
    fig = px.bar(cuisine_order)
    fig.update_layout(yaxis_title=None, xaxis_title=None, showlegend=False)
    st.plotly_chart(fig, use_container_width=True,)

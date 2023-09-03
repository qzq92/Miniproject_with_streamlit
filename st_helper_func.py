"""
Helper functions to facilitate certain streamlit frontend implementation

"""

import streamlit as st
import plotly.express as px
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid
from datetime import datetime

def navbar_edit():
    """Helper function to add text to the top of sidebar.

    Args:
        None
    Returns:
        None
    Raise:
        None
    
    """
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                padding: 1rem 1rem 1rem 1rem;
                font-size: 3rem;
                position: sticky;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Page Navigation";
                display: inline;
            }

            ul {
                font-size: 2rem;
                position: sticky;
            }
        </style>
        
        """,
        unsafe_allow_html=True,
    )
    return None

def adjust_filter_font():
    """Helper function to add text to the top of sidebar.

    Args:
        None
    Returns:
        None
    Raise:
        None
    
    """
    st.markdown(
        """
        <style>
            p {
                font-size: 2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    return None

def add_text_to_filter_pane():
    """Helper function to adjust font and other css values.

    Args:
        None
    Returns:
        None
    Raise:
        None
    
    """
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"]::after {
            content: "Data filters";
            display: inline;
        }

        div[data-baseweb="base-input"] > input { 
            font-size: 1.7rem;
        }

        div[data-baseweb="base-popover"] > li { 
            font-size: 1.7rem;
        }
        div[data-baseweb="select"] > div { 
            font-size: 1.7rem;
        }

        div[data-testid="stMarkdownContainer"] > p { 
            font-size: 2.5rem;
        }

        div[data-testid="stMarkdownContainer"] > p > strong { 
            font-size: 1.7rem;
        }

        div[data-testid="stMetricValue"] > div {
            background-color: navy; 
            font-size: 4rem;
        }

        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1rem;
        }
        
        div[data-testid="stTable"] > table { 
            font-size: 2rem;
        }

        [data-baseweb="select"] {
            margin-top: -3rem;
        }

        [class="stDateInput"] {
            margin-top: -3rem;
            font-size: 4rem;
        }
        </style>
            
        """,
        unsafe_allow_html=True,
    )
    return None
def remove_top_space_canvas():
    """Helper function to remove excess space in canvas section of page.

    Args:
        None
    Returns:
        None
    Raise:
        None

    """
    st.markdown("""
        <style>
                .block-container {
                    padding-top: 1rem; 
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
        </style>
        """, unsafe_allow_html=True
    )
    return None

def gridbuilder_config_setup(df):
    """Helper function that specifies and construct a streamlit AgGrid for a dataframe with fixed settings for the purpose of displaying dataframe content and allowing interaction.

    Args:
        df (dataframe): 
            Dataframe of interest which gridbuilder is to be used on.
    Returns:
        streamlit_gridresponse object for display on streamlit UI 

    Raise:
        None
    """
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=5) #Add pagination
    gb.configure_default_column(selectable=False)
    gb.configure_side_bar() #Add a sidebar
    gb.configure_selection('multiple',
                            use_checkbox=True,
                            groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gridOptions = gb.build()

    grid_response = AgGrid(
        df,
        gridOptions=gridOptions,
        data_return_mode='FILTERED', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=True,
        theme='streamlit', #Add theme color to the table
        enable_enterprise_modules=True,
        domLayout='autoHeight',
        #height=350, 
        width='100%',
        reload_data=False, # Dont reload on each interaction
    )

    #data = grid_response['data']
    #selected = grid_response['selected_rows'] 
    #df = pd.DataFrame(selected) #Pass the selected rows to a new dataframe df

    return grid_response

@st.cache_data(show_spinner=True)
def forecast_default_template_for_no_model():
    """Function that writes a default statement for inference section when no available model can be used for inferencing. 
    
    Args:
        None
    Returns:
        None
    Raise:
        None
    """
    st.write('Forecast not available at the moment with selected model.')

@st.cache_data(show_spinner=True)
def load_data(file_path, parse_date=True):
    """Function that loads data from a given path and utilises st.cache_data decorator (built in cache mechanism) that persists the data across reruns

    Args:
        file_path (string) : Path to file where it is to be loaded into pandas dataframe
        parse_date (bool) : Boolean state to determine whether to parse date
    Returns:
        dataset (dataframe) : Dataframe with or without parsed dates.
    Raise:
        None
    """
    if parse_date:
        custom_date_parser = lambda x: datetime.strptime(x, '%Y-%m-%d')
        dataset = pd.read_csv(file_path,
                            parse_dates=['date'],
                            date_parser=custom_date_parser)
    else:
        dataset = pd.read_csv(file_path)
    return dataset

def add_space_param():
    """Function that add 2 line spaces.
    Args:
        None.
    Returns:
        None.
    Raise:
        None.
    """
    st.text("")
    st.text("")
    return None
import streamlit as st
from st_helper_func import remove_top_space_canvas, navbar_edit
from PIL import Image

# Load image
image = Image.open('img/dataset_info.png')

st.set_page_config(
    layout = "wide",
)


# Remove empty space on top. Use rem for root element font-size referencing (18px)
remove_top_space_canvas()
navbar_edit()

st.title("Mini-project: Food Demand forecasting")

st.header("Problem Statement")

prob_s1, spacer, prob_s2 = st.columns([3,0.2,1.5])

with prob_s1:
    # About dataset
    st.markdown(
        """
        Your client is a meal delivery company which operates in multiple cities. They have various fulfillment centers in these cities for dispatching meal orders to their customers. The client wants you to help these centers with demand forecasting for upcoming weeks so that these centers will plan the stock of raw materials accordingly.

        
        The replenishment of majority of raw materials is done on weekly basis and since the raw material is perishable, the procurement planning is of utmost importance. Secondly, staffing of the centers is also one area wherein accurate demand forecasts are really helpful. Given the following information, the task is to predict the demand for the next 10 weeks (Weeks: 146-155) for the center-meal combinations in the test set:  

        
        - Historical data of demand for a product-center combination (Weeks: 1 to 145)
        - Product(Meal) features such as category, sub-category, current price and discount
        - Information for fulfillment center like center area, city information etc.    
        """
    )

    st.markdown("")
    st.markdown("")
    st.markdown("")
    
    # Team members
    st.header("Team members")
    
    st.markdown(
        """
        - Cerise Choon
        - Goh Jong Ching
        - Loi Xue Zheng
        - Quek Zhi Qiang
        """
    )
    # Css tweaks
    st.markdown(
        """
        <style>
            p {
                font-size: 2.5rem;
            }

            div[data-testid="stMarkdownContainer"] > ul > li { 
                font-size: 2rem;
            }

        </style>
        """
        ,
        unsafe_allow_html=True
    )
    
    with prob_s2:
        st.image(image)
        st.write(
            """
            - [Analytics Vidhya Practice Problem](https://datahack.analyticsvidhya.com/contest/genpact-machine-learning-hackathon-1/#About)
            """ 
        )

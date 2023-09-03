import streamlit as st
from st_helper_func import remove_top_space_canvas, navbar_edit, add_space_param
from PIL import Image
import pandas as pd
# Load image from img subfolder
data_img = Image.open('img/data.jpg')
statistic_img = Image.open('img/df_describe.jpg')
distrb_img = Image.open('img/distribution_plot.jpg')
heatmap_img = Image.open('img/correlation_plot.jpg')
sarima_img = Image.open('img/sarima_rolling_forecast.png')
lstm_img = Image.open('img/lstm.png')
pre_train_img = Image.open('img/pred_train_RNN.jpg')
pre_test_img = Image.open('img/pred_test_RNN.jpg') 
transformer_ts_img = Image.open('img/transformer_timeseries.png') 
transformer_pred = Image.open('img/transformer_prediction.png')
st.set_page_config(
    layout = "wide",
)

# Remove empty space on top. Use rem for root element font-size referencing (18px)
remove_top_space_canvas()
navbar_edit()

st.title("Mini-project: Food Demand forecasting")
st.header("Problem Statement")
st.markdown(
    """
    Your client is a meal delivery company which operates in multiple cities. They have various fulfillment centers in these cities for dispatching meal orders to their customers. The client wants you to help these centers with demand forecasting for upcoming weeks so that these centers will plan the stock of raw materials accordingly.

    The replenishment of majority of raw materials is done on weekly basis and since the raw material is perishable, the procurement planning is of utmost importance. Secondly, staffing of the centers is also one area wherein accurate demand forecasts are really helpful. Given the following information, the task is to predict the demand for the next 10 weeks for the center-meal combinations in the test set:  
    - Historical data of demand for a product-center combination (Weeks: 1 to 145)
    - Product(Meal) features such as category, sub-category, current price and discount
    - Information for fulfillment center like center area, city information etc.

    In this project, we will use 3 different models (SARIMA, LSTM and Transformer) to predict the demand for the next 10 weeks. 
    """
)

add_space_param()

st.header('**Main Deliverable:** Forecasting Dashboard using Streamlit')
st.header('EDA and Data Preprocessing')
st.header('Basic EDA')

col1, col2, col3 = st.columns([1,4,1])
with col1:
    st.write("")
with col2:
    st.image(data_img, use_column_width=True)
with col3:
    st.write("")

st.markdown(
    """
    - Our data has 456,548 row and 15 columns.
    - Our target is “num_orders” and timestamp is in weeks 
    - Numerical columns = 12, Categorical columns = 3 (note that other columns like ID can be categorical as well)
    """
)

add_space_param()

col1, col2, col3 = st.columns([1,4,1])
with col1:
    st.write("")
with col2:
    st.image(statistic_img, use_column_width=True)
    st.image(distrb_img, use_column_width=True)
with col3:
    st.write("")


st.markdown("""
    - Looking at ‘checkout_price’ and ‘base_price’, their values present a very different range and magnitude. Hence, we should standardize the data as the data follows Gaussian distribution. Note that the target is not standardized here.
    """   
)
            
add_space_param()
col1, col2, col3 = st.columns([1,4,1])
with col1:
    st.write("")
with col2:
    st.image(heatmap_img, use_column_width=True)
with col3:
    st.write("")
st.markdown(
    """
    - 'Checkout_price' and 'base_price' has high positive correlation (0.95). Make sense as checkout_price is just base_price with discount, shipping fee, tax.
    - Hence, high correlated features like 'Checkout_price' and 'base_price'. Drop 'Checkout_price' and replace with 'adjusted_price' (difference between checkout and base price).
    - Other observations: 
        -  Both 'homepage_featured' and 'emailer_for_promotion' are positively correlated to num_orders. This makes sense as when there are promotions or items featured, the orders will increase.
        - 'homepage_featured' are positively correlated to 'emailer_for_promotion'. It seems like whenever there is a promotion, the item will likely to be featured on main page.
        - Both 'Checkout_price' and 'base_price' are negatively correlated to 'num_orders' as price increase, the number of orders should decrease. 
    """
)

add_space_param()
st.header('General preprocessing')
st.markdown(
    """
    - Remove useless columns (eg. id)
    - Remove ‘checkout_price’ (high correlation feature) and replace with a price difference column (checkout_price - base_price)
    - Standardize via standscaler for specific columns (eg. exclude ‘num_order’)
    - One-hot encode for categorical features
    - Further preprocessing will be done for each of the model.
    - Note that due to time constraint, we will only be using univariate data (target only) for all models used in this project. This preprocessing pipeline will be used for multivariate time series analysis (future improvement).  
    """
    )

add_space_param()
st.header('**Model 1:** SARIMA model as our baseline model')
st.markdown(
    """
    We began with a SARIMA (Seasonal Autoregressive Integrated Moving Average) model as our baseline model.SARIMA is a popular model for modeling time series data with seasonality.

    The SARIMA model considers the autoregressive and moving average components of a time series, as well as the seasonality of the data. The model is specified by three parameters: p, d, and q, which correspond to the order of the autoregressive, integrated, and moving average components, respectively. Additionally, the model includes seasonal components specified by P, D, and Q, which are similar to the non-seasonal parameters but apply to the seasonal differences and seasonal errors.

    We started by performing a seasonal decomposition of the time series, which showed clear trends and seasonality in the data. We also conducted an Augmented Dickey-Fuller (ADF) test to confirm that the data was not stationary. Next, we analyzed the autocorrelation function (ACF) and partial autocorrelation function (PACF) plots to determine the appropriate orders for the SARIMA model.

    After fitting the SARIMA model to the data, we obtained an RMSE of 95,951.
    """)

col1, col2, col3 = st.columns([1,4,1])
with col1:
    st.write("")
with col2:
    st.image(sarima_img, caption='Template from https://24slides.com/ ')
with col3:
    st.write("")

add_space_param()
st.header('**Model 2:** Long-Short-Term-Memory (LSTM) Model')
st.markdown(
    """
    - RNNs are a type of neural network architecture that is designed to handle sequential data by maintaining a memory of previous inputs. In RNNs, the output of a hidden layer is fed back to the input of the same layer in the next time step. However, the main disadvantage of RNNs is the vanishing gradient problem, where the gradients of the earlier time steps become smaller and smaller, which results in difficulty in training the network and retaining long-term memory.

    - LSTMs are a type of RNN that overcomes the vanishing gradient problem by introducing a memory cell and three gating mechanisms, namely the input gate, forget gate, and output gate. The memory cell helps the network to remember important information over longer time periods, while the gating mechanisms control the flow of information into and out of the cell. The forget gate decides which information should be discarded from the cell and the input gate decides which new information should be added to the cell. The output gate decides the output of the cell at the current time step.
    """
)
col1, col2, col3 = st.columns([1,4,1])
with col1:
    st.write("")

with col2:
    st.image(lstm_img)

with col3:
    st.write("")

add_space_param()
st.header('Training an LSTM model on our food demand data involves the following steps:')

code = """
    class LSTM(nn.Module):
        def __init__(self,input_size, hidden_size, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, 1)
        def forward(self, x):
            x, _ = self.lstm(x)
            x = self.linear(x)
            return x
    """
st.markdown(
    """
    1. Data Preprocessing: Using our 'datapipeline.py', we will preprocess the data. 

    2. Windowing: The data is then windowed, which involves creating overlapping sequences of data points to be used as inputs to the LSTM. We use the 'preprocess_window.py' to generate a normalized train and test torch.tensor dataset. The lookback and lookforward for this project is both 10 respectively (train on past 10 weeks data and predict advance 10 weeks data). 

    3. Model Definition: The LSTM RNN model is defined, specifying the number of layers, the number of neurons in each layer, and the activation function to be used. 

        - The number of LSTM cell used here is 50 and the number of layers is 1.
        - Further hyperparameter tuning can be done by adjusting the hidden_size and num_layers. 
        - nn.Linear() is used here as Linear activation function simply returns the weighted sum of inputs plus a bias term, and is useful for regression problems where the output is a continuous value.

    4. Model Training: The model is then trained on the windowed data using an appropriate optimization algorithm, such as Adam.
        - Further hyperparameter tuning can be done by using different optimizer like RMSProp, Adagrad, SGD. 

    5. Model Evaluation: The trained model is then evaluated on both test and train dataset to measure its performance and to determine if any further modifications to the model are necessary. In this project, we are using RMSE to evaluate our model.
    """)

col1, col2, col3 = st.columns([1,4,1])
with col1:
    st.write("")

with col2:
    st.image(pre_train_img)

with col3:
    st.write("")

col1, col2, col3 = st.columns([1,4,1])
with col1:
    st.write("")

with col2:
    st.image(pre_test_img)

with col3:
    st.write("")

add_space_param()

st.header('**Model 3:** Transformer')
st.markdown(
    """
    Transformers are known for being a state-of-the-art solution to Natural Language Processing (NLP) tasks. However as they are designed to process sequential input data, they can also be used in other tasks such as time series forecasting. 

    The encoder part takes the history of the time series as input while the decoder part predicts the future values in an auto-regressive fashion. The decoder is linked with the encoder using an attention mechanism. This way, the decoder can learn to “attend” to the most useful part of the time series historical values before making a prediction.
    The decoder uses masked self-attention so that the network can’t cheat during training by looking ahead and using future values to predict past values. Positional encoding to provide information about the position of the tokens in the sequence

    The training and inference loops are different as different inputs are required. For training, the ground truth is fed into the decoder. For inference, it is autoregressive, where the first predicted output is fed back into the decoder. 
    """
)
add_space_param()

col1, col2, col3 = st.columns([1,4,1])
with col1:
    st.write("")

with col2:
    st.image(transformer_ts_img, use_column_width=True)

with col3:
    st.write("")

st.markdown(
    """
    When the model is switched to evaluation mode, the predicted values are exactly the same for the entire forecast horizon. In contrast, not switching to evaluation mode returns predicted values which are different from each other. When the model is in evaluation mode, the BatchNorm layer applies normalization using the learned statistics from the training phase, and dropout is disabled. This helps in generating consistent predicted results. 

    In our case, the model does not use any BatchNorm layer, but it does use dropout layers, specifically in the positional encoding layer. Since the positional encoding layer adds a fixed representation to the input sequence, it may cause the model to overfit to the training data. This is because the model may learn to rely too heavily on the positional encoding layer's fixed representation, rather than learning the underlying patterns and relationships within the input sequence. Hence dropout is used in the positional encoding layer.  

    Disabling the dropout layer would effectively disable positional encoding, resulting in constant predicted values. To generate non-constant predicted values, the model is not switched to evaluation mode, and instead a random seed is set to ensure consistency in the predictions. However, the model seems to be underfitting as the predicted values fall within a narrow range, as compared to the actual values which have a much larger range. 
    """
)
col1, col2, col3 = st.columns([1,4,1])
with col1:
    st.write("")

with col2:
    st.image(transformer_pred, use_column_width=True)

with col3:
    st.write("")


add_space_param()
st.header('Comparing RMSE Across Different Models')
st.markdown(
    """
    We evaluated the performance of three different models: SARIMA, LSTM RNN, and Transformer, on a given time series dataset. The table below shows the RMSE values obtained for each model:
    """
)

data = {'Model': ['SARIMA', 'LSTM RNN', 'Transformer'], 'RMSE': ['95,951', '96,430', '86,509']}
sample_df = pd.DataFrame(data)

col1, col2, col3 = st.columns([1,4,1])
with col1:
    st.write("")

with col2:
    st.table(sample_df)

with col3:
    st.write("")

st.markdown(
    """
    Based on these results, we observe that the Transformer model performed the best, with the lowest RMSE value of 86,509.
    """
)

st.header('Future Improvement')
st.markdown(
    """
    - To further improve the performance of the SARIMA model, we can explore hyperparameter tuning techniques. This can involve changing the values of the SARIMA model's order and seasonal_order parameters, or using more advanced techniques such as grid search or Bayesian optimization. Hyperparameter tuning can help us find the optimal set of hyperparameters that can improve the model's performance on the test set.
    
    - To further improve the performance of our neural network model, we can adjust hyperparameters of the model. This includes the number of layers, number of neurons per layer, dropout rate, learning rate, batch size, number of epochs, window size (lookback), sequence length (lookforward) and activation functions.
    
    - So far we only use target variable to train the model. To further improve performance of the model, we can input the other features to train the model as including other features can help the model capture complex relationships between the variables and reduce the impact of noise and outliers in the data.
    """
)

# Css tweaks
st.markdown(
    """
    <style>
        p {
            font-size: 2.5rem;
        }
        [data-testid="stMarkdownContainer"] > p {
            padding-left: 5rem;
            font-size: 2.5rem;
        }
        [data-testid="stMarkdownContainer"] > ul {
            padding-left: 5rem;
        }
        [data-testid="stMarkdownContainer"] > ul > p {
            padding-left: 5rem;
        }

        [data-testid="stCodeBlock"] > div > pre > div > pre > code {
            font-size: 3rem;
        }

        div[data-testid="stMarkdownContainer"] > ol { 
            padding-left: 5rem;
        }

        div[data-testid="stMarkdownContainer"] > ol > li { 
            padding-left: 5rem;
            font-size: 2.5rem;
        }

        div[data-testid="stMarkdownContainer"] > ol > li > ul > li{ 
            font-size: 2.5rem;
        }

        li::marker{ 
            font-size: 2.5rem;
        }

        div[data-testid="stMarkdownContainer"] > ul > li { 
            font-size: 2.5rem;
            padding-left: 5rem;
        }

        div[data-testid="stMarkdownContainer"] > div > ul > li::marker { 
            padding-left: 5rem;
        }
        div[data-testid="stTable"] > table { 
            font-size: 2.5rem;
        }
        div[data-testid="stMarkdownContainer"] > ul > li > ul > li{ 
            font-size: 2.5rem;
        }

    </style>
    """
    ,
    unsafe_allow_html=True
    )

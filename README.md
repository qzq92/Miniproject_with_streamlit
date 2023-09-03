# About repository
This repository is for the 5-day AI Singapore Batch 12 Apprenticeship Programme mini project which is jointly contributed by the following apprentices:

- Goh Jong Ching
- Cerise Choon
- Loi Xue Zheng
- Quek Zhi Qiang

The purpose of this mini project is to allow apprentices to work together and explore and experiment open-source tools available on the internet and utilise them to solve an identified problem statement of interest.

# Problem statement

The derived problem statement below is quoted from the actual link below this section as posted on Analytics Vidhya blog.

Your client is a meal delivery company which operates in multiple cities. They have various fulfillment centers in these cities for dispatching meal orders to their customers. The client wants you to help these centers with demand forecasting for upcoming weeks so that these centers will plan the stock of raw materials accordingly.

The replenishment of majority of raw materials is done on weekly basis and since the raw material is perishable, the procurement planning is of utmost importance. Secondly, staffing of the centers is also one area wherein accurate demand forecasts are really helpful. Given the following information, the task is to predict the demand for the next 10 weeks for the center-meal combinations in the test set:  
- Historical data of demand for a product-center combination (Weeks: 1 to 145)
- Product(Meal) features such as category, sub-category, current price and discount
- Information for fulfillment center like center area, city information etc.

Reference link: [Analytics Vidhya DataHack](https://datahack.analyticsvidhya.com/contest/genpact-machine-learning-hackathon-1/)


## Key deliverables
- The team has tried 3 different models (SARIMA, LSTM and Transformer) to predict the demand for the next 10 weeks as required by the problem statement. 
- To develop a useful forecasting dashboard using suitable open-source tools for showcase and sharing with other apprentices.

## Challenges

The team has encountered some technical challenges in getting LSTM and Transformer models to work on the above problem statement while ensuring the delivery of project within tight deadline. As a result only SARIMA model was successfully experimented.

## Results
- The frontend forecasting dashboard was successfully implemented using Python streamlit library.

## Current status:
- Not in active development since Apr 2023.

# Setup
To run the streamlit service, please clone this repository and install the necessary dependences as provided by environment.yml (using Anaconda distribution) or requirements.txt (Python pip package installer) using either of the following command:

Via Anaconda distribution:

```
conda env create --file=environments.yml

```

Via python pip:

```
pip install -r requirements.txt

```

## Additional stuff to download and place in repository.

Please download `models` and `data` folder from the following link: 
- [data](https://drive.google.com/drive/folders/15bCjCil9vwmI8UCLJdTV84UFZQpPjS61?usp=sharing)
- [models](https://drive.google.com/drive/folders/1U7rj5-p0SWyaZq4-IxWF58ZUHLiGUmlh?usp=sharing)

Place both folders it in the repo folder directly when cloned into directory.

**Please note that only statistical model involving SARIMA is available on the Google Drive link above due to space constraint.**

## Usage

Assuming the necessary dependences has been installed. Please execute the following script to start the streamlit service.

For Windows users: Please run the script `start_streamlit.bat` to start Streamlit service.

For Linux users: Please execute .sh script is developed for Linux users, but not testing has been done. (**Note: This script is added for convenience but has yet to test out on actual machine.**)
# Please edit for reproducibility 

import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os 

def run_datapipeline(data_dir, combine_data_only):
    """Data preprocessing pipeline

    Args:
        data_dir (string)L Folder where downloaded data resides.
        combine_data_only (bool): To perform only combination of data and introduce an arbitrary date as the index based on week information if true. When false, introduce other encoding/feature scaling for other features on top of data combination.
    Returns:
        dataframe: If combine_data_only argument is true, return a single merged dataframe derived from on train/fulfilment and meal info csv files. Otherwise, return a training and testing split of dataframe.
    Raises:

    """
    train_csv_path = os.path.join(os.getcwd(), data_dir, 'train.csv')
    fulfilment_csv_path = os.path.join(os.getcwd(), data_dir, 'fulfilment_center_info.csv')
    meal_info_csv_path = os.path.join(os.getcwd(), data_dir, 'meal_info.csv')


    # Read all 3 datafiles, train, fulfilment and meal info
    try:
        df1 = pd.read_csv(train_csv_path)
        df2 = pd.read_csv(fulfilment_csv_path)
        df3 = pd.read_csv(meal_info_csv_path)
    except IOError:
        raise(IOError)
    
    # 1. Merge all df 
    df = pd.merge(df1, df2, on='center_id',how='left')
    df = pd.merge(df,df3,on='meal_id',how='left')

    # convert weeks to time stamp
    # Calculate the date corresponding to the start of the first week
    start_date = pd.to_datetime('2019-01-28')

    # Calculate the date for each week based on the start date and the week number
    df['date'] = start_date + pd.to_timedelta(df['week'] - 1, unit='W')

    # Set the 'date' column as the index of the DataFrame
    df.set_index('date', inplace=True)

    if combine_data_only:
        # Save merged dataframe
        save_merged_df = os.path.join(data_dir, 'merged.csv')
        df.to_csv(save_merged_df)
        return df
    else:
        # 2. add a add_price_adjustment column
        df['add_price_adjustment'] = df ['checkout_price'] - df ['base_price']

        # category and cuisine very similar. Hence, combine them.
        df['category-cuisine'] = df['category'] +'-'+ df['cuisine']

        # Split train and test first
        train_length = df['week'].nunique()*0.8 #80% train
        df_train = df.loc[df['week'] <= train_length] #week range from 1 to 116
        df_test = df.loc[df['week'] > train_length] #week range from 117 to 145

        # 3. Drop unless columns
        df_train.drop(['id', 'week', 'checkout_price', 'cuisine','category'], axis = 1, inplace =True)
        df_test.drop(['id', 'week', 'checkout_price', 'cuisine','category'], axis = 1, inplace =True)

        #4.Split data into cat and num data for one-hot and standardscalar for df_train
        one_hot_list = ['center_type','category-cuisine']
        normalize_list = ['base_price', 'add_price_adjustment']

        ohe_transformer = OneHotEncoder()
        df_encoded_train = ohe_transformer.fit_transform(df_train[one_hot_list]).toarray()
        df_encoded_df_train = pd.DataFrame(df_encoded_train, columns=ohe_transformer.get_feature_names_out(one_hot_list),index=df_train.index)

        df_encoded_test = ohe_transformer.fit_transform(df_test[one_hot_list]).toarray()
        df_encoded_df_test = pd.DataFrame(df_encoded_test, columns=ohe_transformer.get_feature_names_out(one_hot_list),index=df_test.index)

        # Concatenate the original dataframe and the encoded dataframe
        df_train = pd.concat([df_train, df_encoded_df_train], axis=1)
        df_train.drop(one_hot_list, axis = 1, inplace =True)

        df_test = pd.concat([df_test, df_encoded_df_test], axis=1)
        df_test.drop(one_hot_list, axis = 1, inplace =True)

        # 4.num data for standard scaler
        num_transformer = StandardScaler()
        num_transformer.fit(df_train[normalize_list])

        # transform the selected columns
        scaled_columns_train = num_transformer.transform(df_train[normalize_list])
        scaled_columns_test = num_transformer.transform(df_test[normalize_list])

        # create a new DataFrame with the scaled columns
        df_train[normalize_list] = scaled_columns_train
        df_test[normalize_list] = scaled_columns_test

        return df_train , df_test

import pandas as pd


def merge_messages_and_categories(df_messages, df_categories):
    temp = pd.merge(df_messages, df_categories, how="left", on="id")
    return temp


def expand_categories(df_categories):
    temp = df_categories.copy()
    temp = temp["categories"].str.split(";", expand=True)
    temp.columns = [s.split("-")[0] for s in temp.iloc[0]]
    return temp


def binarize_categories(df_categories):
    temp = df_categories.copy()
    for column in temp:
        temp[column] = temp[column].str[-1]
        temp[column] = temp[column].astype(int)
    return temp


def replace_categories_columns(df_data, df_categories_new):
    temp = df_data.copy()
    temp.drop(columns=["categories"], inplace=True)
    temp = pd.concat([temp, df_categories_new], axis=1)
    return temp


def drop_duplicates(df_data):
    temp = df_data.drop_duplicates()
    return temp


def drop_columns_with_too_many_nan_values(df_data, drop_threshold=0.5):
    temp = df_data.copy()
    temp = temp.loc[:, temp.isnull().mean() < drop_threshold]
    temp = temp.reset_index(drop=True)
    return temp


def drop_rows_with_nan_values(df_data):
    temp = df_data.copy()
    temp = temp.dropna()
    return temp


def remove_single_value_categories(df_data):
    temp = df_data.copy()
    criteria = [c for c in temp.columns if temp[c].nunique() > 1]
    temp = temp[criteria]
    return temp

import pandas as pd


def merge_messages_and_categories(df_messages, df_categories):
    """Merges two messages and categories into one dataframe on shared "id" column.

    Parameters:
    -----------
    df_messages: pd.DataFrame
        Dataframe containing disaster messages in raw format.

    df_categories: pd.DataFrame
         Dataframe containing categories of disaster messages in raw format.
    """
    temp = pd.merge(df_messages, df_categories, how="left", on="id")
    return temp


def expand_categories(df_categories):
    """Splits category names which are stacked into single string. Creates separate column for each category. Works on
    a copy.

    Parameters:
    -----------
    df_categories: pd.DataFrame
        Dataframe containing categories of disaster messages in raw format.
    """
    temp = df_categories.copy()
    temp = temp["categories"].str.split(";", expand=True)
    temp.columns = ["category_" + s.split("-")[0] for s in temp.iloc[0]]
    return temp


def categories_to_integers(df_categories):
    """Moves category names from values to each category column. Changes string values to integers. Works on a copy.

    Parameters:
    -----------
    df_categories: pd.DataFrame
        Dataframe containing expanded categories thorough columns with string values.
    """
    temp = df_categories.copy()
    for column in temp:
        temp[column] = temp[column].str[-1]
        temp[column] = temp[column].astype(int)
    return temp


def replace_categories_columns(df_data, df_categories_new):
    """Removes "categories" columns from dataframe and replaces it with other dataframe. Works on a copy.

    Parameters:
    -----------
    df_data: pd.DataFrame
        Dataframe which is a result of merging messages and categories. Whole data.

    df_categories_new: pd.DataFrame
        Dataframe containing wrangled categories.
    """
    temp = df_data.copy()
    temp.drop(columns=["categories"], inplace=True)
    temp = pd.concat([temp, df_categories_new], axis=1)
    return temp


def drop_duplicates(df):
    """Removes duplicates from sent dataframe. Returns a copy.

    Parameters:
    -----------
    df: pd.DataFrame
        Dataframe from which duplicates will be removed.
    """
    temp = df.drop_duplicates()
    return temp


def drop_columns_with_too_many_nan_values(df, drop_threshold=0.5):
    """Removes columns which nan value percentage is larger than threshold. Works on a copy.

    Parameters:
    -----------
    df: pd.DataFrame
        Dataframe from which columns should be removed.

    drop_threshold: float
        Percentage threshold based on which columns will be dropped.
    """
    temp = df.copy()
    temp = temp.loc[:, temp.isnull().mean() < drop_threshold]
    temp = temp.reset_index(drop=True)
    return temp


def drop_rows_with_nan_values(df):
    """Removes rows containing nan values. Works on a copy.

    Parameters:
    -----------
    df: pd.DataFrame
        Dataframe from which rows will be removed.
    """
    temp = df.copy()
    temp = temp.dropna()
    return temp


def remove_single_value_categories(df_data):
    """Removes columns containing only one value. Works on a copy.

    Parameters:
    -----------
    df: pd.DataFrame
        Dataframe which is a result of merging messages and categories. Whole data.
    """
    temp = df_data.copy()
    criteria = [c for c in temp.columns if temp[c].nunique() > 1]
    temp = temp[criteria]
    return temp

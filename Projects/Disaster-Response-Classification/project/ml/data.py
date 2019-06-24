from sklearn.model_selection import train_test_split


def prepare_datasets(df, test_ratio=0.2):
    """Slices dataframe into inputs and target columns. Columns are sliced and shuffled again via train_test_split
    function of scikit-learn. Function returns train and test inputs as well as train and test targets sliced in ratio
    set by 'test_ratio' parameter.

    Parameters:
    -----------
    df: pd.DataFrame
        Dataframe to be sliced in to X_train, X_test, Y_train, Y_test parts.
    """
    X = df["message"]
    Y = df[[col for col in df.columns if "category" in col]]
    return train_test_split(X, Y, test_size=test_ratio)

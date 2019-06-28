from plotly.graph_objs import Bar


def add_category_count_graph(df):
    """Returns jsonified plotly barplot of message sources grouped and counted by category.

    Parameters:
    -----------
    df: pd.DataFrame
        Dataframe with preprocessed data of messages and categories.
    """
    category_counts = df[[c for c in df.columns if "category" in c]].sum(axis=0)
    category_names = [c[9:] for c in category_counts.index]
    category_counts = category_counts.values

    return {
        "data": [
            Bar(
                x=category_names,
                y=category_counts
            )
        ],

        "layout": {
            "title": "Number of messages per category",
            "yaxis": {
                "title": "Count"
            },
            'height': 600,
            'margin': dict(b=200, pad=4),
        }
    }


def add_source_count_graph(df):
    """Returns jsonified plotly barplot of message sources grouped and counted by genre.

    Parameters:
    -----------
    df: pd.DataFrame
        Dataframe with preprocessed data of messages and categories.
    """
    genre_counts = df["genre"].value_counts()
    genre_names = genre_counts.index
    genre_counts = genre_counts.values

    return {
        "data": [
            Bar(
                x=genre_names,
                y=genre_counts
            )
        ],

        "layout": {
            "title": "Source of gathered messages",
            "yaxis": {
                "title": "Count"
            },
            'height': 600,
            'margin': dict(b=200, pad=4),
        }
    }

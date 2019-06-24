import pandas as pd
from sqlalchemy import create_engine


def get_database_engine(path):
    """Creates sqlalchemy.engine.base.Engine object via sqlalchemy library.

    Parameters:
    -----------
    path: str
        Path where .db file will be created.
    """
    return create_engine("sqlite:///{}".format(path))


def df_to_sql_table(df, table_name, engine):
    """Dumps dataframe to .db file at specified table name.

    Parameters:
    -----------
    df: pd.DataFrame
        Dataframe containing data which should be saved.

    table_name: str
        Name of sql table to which data will be saved.

    engine: sqlalchemy.engine.base.Engine
        Wrapper (engine) created via sqlalchemy lib over .db file which allows interaction with database.
    """
    df.to_sql(table_name, engine, index=False)


def sql_table_to_df(table_name, engine):
    return pd.read_sql_table(table_name, engine)

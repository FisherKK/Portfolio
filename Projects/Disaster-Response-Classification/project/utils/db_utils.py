from sqlalchemy import create_engine


def get_database_engine(path):
    return create_engine("sqlite:///{}".format(path))


def df_to_sql_table(df, table_name, engine):
    df.to_sql(table_name, engine, index=False)

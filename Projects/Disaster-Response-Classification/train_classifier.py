import argparse

from project.config import (
    DEFAULT_DB_OUTPUT_FILE_PATH,
    DEFAULT_MODEL_OUTPUT_FILE_PATH,
    SQL_TABLE_NAME
)

from project.utils.db_utils import (
    get_database_engine,
    sql_table_to_df
)

from project.utils.model_utils import save_model

from project.preprocessing.feature_engineering import tokenize

from project.ml.data import prepare_datasets
from project.ml.model import get_lgbm_model_pipeline
from project.ml.metric import get_classification_result

parser = argparse.ArgumentParser()
parser.add_argument("--db_path",
                    help="Absolute path for .db file from where data should be loaded.",
                    default=DEFAULT_DB_OUTPUT_FILE_PATH,
                    type=str)

parser.add_argument("--model_path",
                    help="Absolute path to where model should be saved.",
                    default=DEFAULT_MODEL_OUTPUT_FILE_PATH,
                    type=str)


def train_and_save_model():
    """Function which loads data from .db object passed via arguments from command line or default directory. Data is
    preprocessed with nlp pipeline which tokenizes, cleans, lematizes, stemes the text. Then CountVectorizer is used and
    finally Tfidif performed. Data is divided into train/test datasets and used for grid search hyperparameter tuning
    with cross validation on LGBM Classifier model. Results are being displayed to user and model is saved in directory
    passed via args or default one.
    """
    args = parser.parse_args()

    df_data = sql_table_to_df(SQL_TABLE_NAME, get_database_engine(args.db_path))
    X_train, X_test, Y_train, Y_test = prepare_datasets(df_data, test_ratio=0.2)

    model_cv = get_lgbm_model_pipeline(tokenize_function=tokenize)
    model_cv.fit(X_train, Y_train)

    model = model_cv.best_estimator_

    train_pred = model.predict(X_train)
    train_score = get_classification_result(Y_train, train_pred, [c for c in df_data.columns if "category" in c])
    print("Train score:\n{}".format(train_score.head()))

    test_pred = model.predict(X_test)
    test_score = get_classification_result(Y_test, test_pred, [c for c in df_data.columns if "category" in c])
    print("Test score:\n{}".format(test_score.head()))

    save_model(args.model_path, model)
    print("Saved model to '{}' directory.".format(args.model_path))


if __name__ == '__main__':
    train_and_save_model()

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

import lightgbm as lgbm

from project.config import RANDOM_SEED


def get_lgbm_model_pipeline(**kwargs):
    """Returns a scikit-learn pipeline containing CountVectorizer and TfidfTransformer and LGBMClassifier wrapped onto
    GridSearchCV object.

    Parameters:
    -----------
    kwargs: **kwargs
        Parameters passed by name. Parameter "tokenize_function" is required to fill CountVectorizer.
    """
    model_lgbm = Pipeline([
        ("text_pipeline", Pipeline([
            ("vect", CountVectorizer(tokenizer=kwargs.get("tokenize_function"))),
            ("tfidf", TfidfTransformer())
        ])),
        ("clf", MultiOutputClassifier(lgbm.LGBMClassifier()))
    ])

    parameters = {
        # "text_pipeline__vect__ngram_range": ((1, 1), (1, 2)),
        # "text_pipeline__vect__max_df": (0.5, 0.75, 1.0),
        # "text_pipeline__vect__max_features": (None, 5000, 10000),
        # "clf__estimator__num_leaves": [5, 10, 15, 20],
        # "clf__estimator__max_depth": [-1, 5, 15, 25],
        "clf__estimator__seed": [RANDOM_SEED]
    }

    return GridSearchCV(model_lgbm, param_grid=parameters, verbose=2)

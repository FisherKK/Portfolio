from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.layers import (
    Activation,
    Dropout,
    Dense
)

from tensorflow.keras.models import (
    Sequential,
    Model

)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from project.config import GLOBAL_SEED


def train_xgboost_model(train_data, val_data, test_data):
    train_samples, train_targets = train_data
    val_samples, val_targets = val_data
    test_samples, test_targets = test_data

    params = {
        "max_depth": 5,
        "eta": 0.275,
        "subsample": 0.95,
        "reg_lambda": 0.1,
        "reg_alpha": 0.1,
        "objective": "multi:softmax",
        "predictor": "cpu_predictor",
        "booster": "gbtree",
        "tree_method": "hist",
        "verbosity": 0,
        "n_jobs": -1,
        "random_state": GLOBAL_SEED
    }

    model = XGBClassifier(**params)
    model.fit(
        train_samples, train_targets,
        eval_set=[(train_samples, train_targets), (val_samples, val_targets)],
        early_stopping_rounds=10,
        eval_metric=["merror"],
        verbose=True
    )

    train_pred = model.predict(train_samples)
    train_accuracy = accuracy_score(train_pred, train_targets)

    test_pred = model.predict(test_samples)
    test_accuracy = accuracy_score(test_pred, test_targets)

    print("\nModel results:")
    print("\t train | accuracy: {}".format(train_accuracy))
    print("\t  test | accuracy: {}\n".format(test_accuracy))

    return model


def train_simple_mlp_model(train_data, val_data, test_data):
    train_samples, train_targets_ohe = train_data
    val_samples, val_targets_ohe = val_data
    test_samples, test_targets_ohe = test_data

    model = Sequential()
    model.add(Dense(512, input_shape=(train_samples.shape[1],)))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))

    model.add(Dense(384))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))

    model.add(Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=Adam(0.005), metrics=["accuracy"])
    model.summary()

    model.fit(
        train_samples, train_targets_ohe,
        batch_size=512,
        epochs=100,
        verbose=2,
        callbacks=[EarlyStopping(patience=10)],
        validation_data=(val_samples, val_targets_ohe)
    )

    train_result = model.evaluate(train_samples, train_targets_ohe, verbose=0)
    val_result = model.evaluate(val_samples, val_targets_ohe, verbose=0)
    test_result = model.evaluate(test_samples, test_targets_ohe, verbose=0)

    print("\nModel results:")
    print("\t train | loss: {}, accuracy: {}".format(*train_result))
    print("\t   val | loss: {}, accuracy: {}".format(*val_result))
    print("\t  test | loss: {}, accuracy: {}\n".format(*test_result))

    return model

import os

from project.config import MODEL_DIR


def save_mlp_model(model, name):
    filepath = os.path.join(MODEL_DIR, "{}.h5".format(name))
    model.save(filepath)
    print("Successfully saved model to '{}' filepath.".format(filepath))


def save_xgboost_model(model, name):
    filepath = os.path.join(MODEL_DIR, "{}".format(name))
    model.save_model(filepath)
    print("Successfully saved model to '{}' filepath.".format(filepath))

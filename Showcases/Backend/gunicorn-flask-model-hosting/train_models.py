from project.data.dataset import load_mnist_data
from project.config.model import MODEL_PROPERTIES


def _app():
    """For each model set in MODEL_PROPERTIES it loads preprocessed MNIST data, builds model and saves is in MODEL_DIR
    directory.
    """
    for model_key, properties in MODEL_PROPERTIES.items():
        print("\n-------- Building '{}' model.".format(model_key))

        build_model_func = properties["build_function"]
        save_model_func = properties["save_function"]
        use_ohe = properties["ohe"]

        train_data, val_data, test_data = load_mnist_data(ohe=use_ohe)
        model = build_model_func(train_data, val_data, test_data)
        save_model_func(model, model_key)


if __name__ == "__main__":
    _app()

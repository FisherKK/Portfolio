from project.data.loading import load_mnist_data
from project.ml.building import (
    train_simple_mlp_model,
    train_xgboost_model,
)

from project.ml.saving import (
    save_xgboost_model,
    save_mlp_model
)

build_config = {
    "mlp": {
        "build_function": train_simple_mlp_model,
        "save_function": save_mlp_model,
        "ohe": True
    },
    "xgboost": {
        "build_function": train_xgboost_model,
        "save_function": save_xgboost_model,
        "ohe": False
    },
}

if __name__ == "__main__":
    for model_key, model_params in build_config.items():
        print("\n-------- Building '{}' model.".format(model_key))

        build_model_func = model_params["build_function"]
        save_model_func = model_params["save_function"]
        use_ohe = model_params["ohe"]

        train_data, val_data, test_data = load_mnist_data(ohe=use_ohe)
        model = build_model_func(train_data, val_data, test_data)
        save_model_func(model, model_key)

from project.ml.building import (
    train_simple_mlp_model,
    train_xgboost_model,
)

from project.ml.saving import (
    save_xgboost_model,
    save_mlp_model
)

from project.ml.loading import (
    load_xgboost_model,
    load_mlp_model
)

from project.ml.inference import (
    predict_with_xgboost_model,
    predict_with_mlp_model
)

MODEL_TYPE_MLP = "mlp"
MODEL_TYPE_XGBOOST = "xgboost"

MODEL_PROPERTIES = {
    MODEL_TYPE_MLP: {
        "build_function": train_simple_mlp_model,
        "save_function": save_mlp_model,
        "load_function": load_mlp_model,
        "prediction_function": predict_with_mlp_model,
        "ohe": True
    },
    MODEL_TYPE_XGBOOST: {
        "build_function": train_xgboost_model,
        "save_function": save_xgboost_model,
        "load_function": load_xgboost_model,
        "prediction_function": predict_with_xgboost_model,
        "ohe": False
    }
}

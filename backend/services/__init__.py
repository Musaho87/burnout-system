# Add training module to services
from .training import (
    train_from_csv,
    safe_train_from_csv,
    get_active_model,
    get_all_models,
    activate_model,
    predict_burnout,
    get_model_statistics,
    delete_model
)
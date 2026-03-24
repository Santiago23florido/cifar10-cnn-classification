from .inventory import (
    get_experimental_conditions,
    get_hyperparameter_inventory,
    get_non_hyperparameter_constants,
    get_section5_audit,
    get_trainable_parameter_summary,
)


def get_section5_payload() -> dict:
    return {
        "audit": get_section5_audit(),
        "trainable_summary": get_trainable_parameter_summary(),
        "hyperparameters": get_hyperparameter_inventory(),
        "experimental_conditions": get_experimental_conditions(),
        "excluded": get_non_hyperparameter_constants(),
    }

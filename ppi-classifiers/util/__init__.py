# Andrew Chung, hc893; 6/20/25

import importlib

__all__ = [
    "manual_resample_training_data",
    "analyze_predictions",
    "create_ratioed_test_set_oversample_neg",
    "plot_roc_pr"

    #"""TODO"""
]

_SYMBOL_TO_MODULE = {
    "manual_resample_training_data": ("classifier_util", "manual_resample_training_data"),
    "analyze_predictions": ("classifier_util", "analyze_predictions"),
    "create_ratioed_test_set_oversample_neg":   ("classifier_util",   "create_ratioed_test_set_oversample_neg"),
    "plot_roc_pr": ("classifier_util",   "plot_roc_pr")

    #"""TODO"""
}

def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        mod_name, attr = _SYMBOL_TO_MODULE[name]
        module = importlib.import_module(f".{mod_name}", __name__)
        return getattr(module, attr)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return __all__

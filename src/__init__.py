"""
Package src - Module de chargement des modèles ML
"""

from .model_loader import (
    load_models_for_target,
    get_available_targets,
    get_model_info
)

from .preprocessor import (
    load_and_preprocess_data,
    prepare_features_for_model
)

__all__ = [
    'load_models_for_target',
    'get_available_targets',
    'get_model_info',
    'load_and_preprocess_data',
    'prepare_features_for_model'
]

__version__ = '2.0.0'
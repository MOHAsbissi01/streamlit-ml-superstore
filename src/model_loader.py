"""
Module pour charger les modèles ML target-specifiques
"""

import joblib
import os
import pickle
from pathlib import Path

# Chemin vers le dossier models
MODELS_DIR = Path("models")

# Mapping des modèles target-specifiques
TARGET_MODELS = {
    'Sales': {
        "Linear Regression": "linear_regression_sales.pkl",
        "Random Forest": "random_forest_sales.pkl",
        "Gradient Boosting": "gradient_boosting_sales.pkl",
        "LightGBM": "lightgbm_sales.pkl",
        "Voting Regressor": "voting_regressor_sales.pkl",
    },
    'Profit': {
        "Linear Regression": "linear_regression_profit.pkl",
        "Random Forest": "random_forest_profit.pkl",
        "Gradient Boosting": "gradient_boosting_profit.pkl",
        "LightGBM": "lightgbm_profit.pkl",
        "Voting Regressor": "voting_regressor_profit.pkl",
    },
    'Quantity': {
        "Linear Regression": "linear_regression_quantity.pkl",
        "Random Forest": "random_forest_quantity.pkl",
        "Gradient Boosting": "gradient_boosting_quantity.pkl",
        "LightGBM": "lightgbm_quantity.pkl",
        "Voting Regressor": "voting_regressor_quantity.pkl",
    }
}


def load_models_for_target(target):
    """
    Charge tous les modèles pour un target spécifique
    
    Args:
        target (str): Le nom du target ('Sales', 'Profit', ou 'Quantity')
    
    Returns:
        dict: Dictionnaire {nom_modele: modele_chargé}
    """
    models = {}
    
    if target not in TARGET_MODELS:
        print(f"Target '{target}' non reconnu. Targets disponibles: {list(TARGET_MODELS.keys())}")
        return models
    
    model_files = TARGET_MODELS[target]
    
    print(f"\nChargement des modèles pour target '{target}'...")
    print("="*60)
    
    loaded_count = 0
    
    for model_name, filename in model_files.items():
        filepath = MODELS_DIR / filename
        
        if not filepath.exists():
            print(f"  {model_name}: fichier non trouvé ({filename})")
            continue
        
        try:
            model = joblib.load(filepath)
            
            if hasattr(model, 'predict'):
                models[model_name] = model
                print(f"  {model_name}: chargé")
                loaded_count += 1
            else:
                print(f"  {model_name}: pas un modèle valide")
                
        except Exception as e:
            print(f"  {model_name}: erreur de chargement - {str(e)[:50]}")
    
    print("="*60)
    print(f"Résumé: {loaded_count} modèles chargés pour {target}")
    
    return models


def get_available_targets():
    """
    Retourne la liste des targets disponibles
    
    Returns:
        list: Liste des targets
    """
    return list(TARGET_MODELS.keys())


def get_model_info(model):
    """
    Obtenir des informations sur un modèle chargé
    
    Args:
        model: Le modèle
    
    Returns:
        dict: Informations sur le modèle
    """
    info = {
        "type": type(model).__name__,
        "module": type(model).__module__,
    }
    
    # Essayer d'obtenir les paramètres
    try:
        info["params"] = model.get_params()
    except:
        info["params"] = "Non disponible"
    
    # Essayer d'obtenir les features
    try:
        if hasattr(model, 'feature_names_in_'):
            info["features"] = list(model.feature_names_in_)
        elif hasattr(model, 'n_features_in_'):
            info["n_features"] = model.n_features_in_
    except:
        pass
    
    return info


# Exemple d'utilisation
if __name__ == "__main__":
    print("="*60)
    print("TEST DU MODULE MODEL_LOADER")
    print("="*60)
    
    # Targets disponibles
    print("\nTargets disponibles:")
    for target in get_available_targets():
        print(f"  - {target}")
    
    # Charger les modèles pour chaque target
    for target in get_available_targets():
        models = load_models_for_target(target)
        print(f"\n{target}: {len(models)} modèles chargés")
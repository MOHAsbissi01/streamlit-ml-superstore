"""
Prétraitement des données pour correspondre aux modèles entraînés
"""

import pandas as pd
import numpy as np


def load_and_preprocess_data(csv_path="data/Global_Superstore_FIXED.csv"):
    """
    Charge et prétraite le CSV pour correspondre aux modèles entraînés
    
    Returns:
        pd.DataFrame: DataFrame prétraité avec toutes les features encodées
    """
    # Charger le CSV
    df = pd.read_csv(csv_path)
    
    print(f"\n📥 CSV chargé: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print(f"Colonnes: {list(df.columns)}\n")
    
    # ===================================
    # 1. IDENTIFIER les types de colonnes
    # ===================================
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"📊 {len(numeric_cols)} colonnes numériques")
    print(f"📝 {len(text_cols)} colonnes textuelles")
    
    # Afficher la cardinalité des colonnes textuelles
    print(f"\n🔍 Cardinalité des colonnes textuelles:")
    for col in text_cols:
        n_unique = df[col].nunique()
        print(f"   {col:30s} → {n_unique:6,} valeurs uniques")
    
    # ===================================
    # 2. SUPPRIMER toutes les colonnes textuelles SAUF les catégorielles à faible cardinalité
    # ===================================
    
    # Colonnes catégorielles valides (< 50 valeurs uniques)
    valid_categorical = []
    for col in text_cols:
        if df[col].nunique() < 50:
            valid_categorical.append(col)
    
    print(f"\n✅ Colonnes catégorielles valides (<50 valeurs): {valid_categorical}")
    
    # Colonnes à supprimer
    columns_to_drop = [col for col in text_cols if col not in valid_categorical]
    
    print(f"\n🗑️ Suppression de {len(columns_to_drop)} colonnes textuelles:")
    for col in columns_to_drop:
        print(f"   ✂️ {col}")
    
    df = df.drop(columns=columns_to_drop)
    
    print(f"\n📊 Après suppression: {df.shape[1]} colonnes")
    
    # ===================================
    # 3. One-Hot Encoding UNIQUEMENT des colonnes catégorielles valides
    # ===================================
    
    if valid_categorical:
        print(f"\n🔄 Encodage de {len(valid_categorical)} colonnes catégorielles...")
        
        for col in valid_categorical:
            print(f"   📊 {col}: {df[col].nunique()} valeurs → {df[col].nunique()} colonnes")
        
        # Créer les dummies
        df_encoded = pd.get_dummies(df, columns=valid_categorical, drop_first=False)
        
        print(f"\n✅ Après encodage: {df_encoded.shape[1]} colonnes")
    else:
        print("\n⚠️ Aucune colonne catégorielle à encoder")
        df_encoded = df
    
    # ===================================
    # 4. Créer Avg_Unit_Price si manquante
    # ===================================
    
    if 'Avg_Unit_Price' not in df_encoded.columns:
        if 'Sales' in df_encoded.columns and 'Quantity' in df_encoded.columns:
            df_encoded['Avg_Unit_Price'] = np.where(
                df_encoded['Quantity'] != 0,
                df_encoded['Sales'] / df_encoded['Quantity'],
                0
            )
            print("✅ Colonne 'Avg_Unit_Price' créée")
    
    # ===================================
    # 5. Résultat final
    # ===================================
    
    numeric_cols_final = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\n📊 Résultat final:")
    print(f"   - {df_encoded.shape[0]:,} lignes")
    print(f"   - {df_encoded.shape[1]} colonnes totales")
    print(f"   - {len(numeric_cols_final)} colonnes numériques")
    print(f"\n✅ Colonnes finales: {list(df_encoded.columns)[:20]}...\n")
    
    return df_encoded


def prepare_features_for_model(df, target_col, model):
    """
    Prépare les features pour un modèle spécifique
    
    Args:
        df: DataFrame prétraité
        target_col: Nom de la colonne cible
        model: Modèle ML
    
    Returns:
        X, y, missing_features
    """
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible '{target_col}' introuvable!")
    
    y = df[target_col]
    
    # Si le modèle a des features spécifiques
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        
        # CRITICAL CHECK: Si la target est dans les features attendues,
        # cela signifie que le modèle a été entraîné pour prédire une AUTRE variable
        if target_col in expected_features:
            # Trouver quelle était la vraie target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            possible_original_targets = [col for col in numeric_cols if col not in expected_features]
            
            error_msg = f"\n❌ Ce modèle ne peut PAS prédire '{target_col}'"
            error_msg += f"\n\n🔍 Raison: Le modèle attend '{target_col}' comme FEATURE (entrée), pas comme TARGET (sortie)."
            
            if possible_original_targets:
                error_msg += f"\n\n💡 Ce modèle a probablement été entraîné pour prédire: {possible_original_targets}"
                error_msg += f"\n\n✅ Solution: Changez la variable cible vers l'une de ces options:"
                for target in possible_original_targets[:3]:
                    error_msg += f"\n   • {target}"
            else:
                error_msg += f"\n\n💡 Utilisez un modèle différent ou réentraînez les modèles."
            
            print(error_msg)
            return None, None, [target_col]  # Retourner le target comme "missing" pour déclencher l'erreur
        
        # Vérifier les features manquantes
        missing = [f for f in expected_features if f not in df.columns]
        
        if missing:
            print(f"\n⚠️ {len(missing)} features manquantes pour ce modèle")
            
            # Catégoriser les features manquantes
            categorical_missing = [f for f in missing if any(key in f for key in ['City', 'State', 'Country', 'Product Name', 'Customer ID'])]
            other_missing = [f for f in missing if f not in categorical_missing]
            
            if categorical_missing:
                print(f"\n❌ Ce modèle nécessite des colonnes géographiques/catégorielles qui ont été supprimées:")
                print(f"   {categorical_missing[:5]}...")
                print(f"\n💡 Ces modèles ont été entraînés avec des données différentes.")
                print(f"   Utilisez plutôt: 'Linear Regression', 'Random Forest Champion', ou 'LightGBM Robust'")
                return None, None, missing
            
            # Pour les autres features manquantes, les créer avec des zéros
            if other_missing:
                print(f"\n🔧 Ajout de {len(other_missing)} features manquantes avec valeur 0:")
                if len(other_missing) <= 10:
                    for feat in other_missing:
                        print(f"   + {feat}")
                else:
                    for feat in other_missing[:5]:
                        print(f"   + {feat}")
                    print(f"   ... et {len(other_missing) - 5} autres")
                
                # Créer une copie du DataFrame et ajouter les colonnes manquantes
                df_copy = df.copy()
                for feat in other_missing:
                    df_copy[feat] = 0
                
                # Sélectionner les features dans le bon ordre
                X = df_copy[expected_features]
                print(f"✅ Features alignées: {X.shape[1]} colonnes")
                return X, y, []
        
        # Sélectionner les features dans le bon ordre
        X = df[expected_features]
    else:
        # Utiliser toutes les colonnes numériques sauf la cible
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col]
        X = df[feature_cols]
    
    return X, y, []





# Test du module
if __name__ == "__main__":
    print("="*60)
    print("🧪 TEST DU PRÉTRAITEMENT")
    print("="*60)
    
    df = load_and_preprocess_data()
    
    print("\n📊 Aperçu des 5 premières lignes:")
    print(df.head())
    
    print("\n✅ Test terminé!")
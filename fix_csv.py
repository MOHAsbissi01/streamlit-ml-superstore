"""
Script pour corriger le CSV avec guillemets complexes
"""

import pandas as pd
from pathlib import Path
import csv
import re

def fix_csv_advanced(input_path, output_path):
    """
    Corrige le CSV avec guillemets complexes en utilisant le module csv de Python
    """
    print("="*60)
    print("🔧 CORRECTION AVANCÉE DU CSV")
    print("="*60)
    
    print(f"\n📥 Lecture du fichier: {input_path}")
    
    # Lire avec le module csv natif de Python qui gère mieux les guillemets
    rows = []
    
    with open(input_path, 'r', encoding='utf-8', newline='') as f:
        # Configuration du reader CSV
        # Le CSV a des lignes encapsulées dans des guillemets avec des guillemets doubles internes
        reader = csv.reader(f, 
                          delimiter=',',
                          quotechar='"',
                          doublequote=True,
                          skipinitialspace=False)
        
        for i, row in enumerate(reader):
            rows.append(row)
            
            if i % 10000 == 0 and i > 0:
                print(f"   ⏳ {i} lignes lues...")
    
    print(f"✅ {len(rows)} lignes lues avec succès")
    
    # Vérifier le nombre de colonnes
    if rows:
        header = rows[0]
        n_cols = len(header)
        print(f"\n📊 Structure détectée:")
        print(f"   - Colonnes: {n_cols}")
        print(f"   - Header: {', '.join(header[:5])}...")
        
        # Vérifier la cohérence
        inconsistent_lines = []
        for i, row in enumerate(rows[1:], start=1):
            if len(row) != n_cols:
                inconsistent_lines.append((i, len(row)))
                if len(inconsistent_lines) <= 5:  # Afficher les 5 premières
                    print(f"   ⚠️  Ligne {i}: {len(row)} colonnes au lieu de {n_cols}")
        
        if inconsistent_lines:
            print(f"\n⚠️  {len(inconsistent_lines)} lignes avec nombre de colonnes incohérent")
            print("🔄 Tentative de correction...")
            
            # Garder seulement les lignes avec le bon nombre de colonnes
            cleaned_rows = [rows[0]]  # Header
            for i, row in enumerate(rows[1:], start=1):
                if len(row) == n_cols:
                    cleaned_rows.append(row)
            
            print(f"✅ {len(cleaned_rows)-1} lignes valides conservées")
            rows = cleaned_rows
    
    # Créer un DataFrame
    df = pd.DataFrame(rows[1:], columns=rows[0])
    
    print(f"\n✅ DataFrame créé:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Colonnes: {df.shape[1]}")
    
    # Identifier et convertir les colonnes numériques
    print(f"\n🔄 Conversion des colonnes numériques...")
    
    numeric_converted = 0
    for col in df.columns:
        try:
            # Essayer de convertir en numérique
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].notna().sum() / len(df) > 0.5:  # Si plus de 50% sont numériques
                numeric_converted += 1
        except:
            pass
    
    print(f"✅ {numeric_converted} colonnes converties en numérique")
    
    # Afficher les types finaux
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n📊 Types de colonnes:")
    print(f"   - Numériques ({len(numeric_cols)}): {numeric_cols[:5]}")
    print(f"   - Textuelles ({len(text_cols)}): {text_cols[:5]}")
    
    # Sauvegarder
    if output_path:
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n💾 Fichier sauvegardé: {output_path}")
    
    # Afficher un aperçu
    print(f"\n📊 Aperçu des premières lignes:")
    print(df.head())
    
    print(f"\n📊 Info du DataFrame:")
    print(df.info())
    
    return df


def create_training_script_v2(df):
    """
    Crée un script d'entraînement optimisé
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Détecter la target
    target_candidates = ['Profit', 'Sales', 'Revenue', 'Price']
    target = None
    
    for candidate in target_candidates:
        if candidate in numeric_cols:
            target = candidate
            break
    
    if target is None:
        target = numeric_cols[0] if numeric_cols else 'Profit'
    
    features = [col for col in numeric_cols if col != target]
    
    code = f'''"""
Script d\'entraînement des modèles ML
Généré automatiquement - Global Superstore
"""

import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Configuration
os.makedirs("models", exist_ok=True)

print("="*70)
print("🚀 ENTRAÎNEMENT DES MODÈLES - GLOBAL SUPERSTORE")
print("="*70)

# 1. Charger les données
print("\\n📥 Chargement des données...")
df = pd.read_csv("data/Global_Superstore_FIXED.csv")
print(f"✅ Dataset: {{df.shape[0]:,}} lignes × {{df.shape[1]}} colonnes")

# 2. Configuration
TARGET = '{target}'
FEATURES = {features}

print(f"\\n🎯 Configuration:")
print(f"   Target: {{TARGET}}")
print(f"   Features: {{len(FEATURES)}}")

# 3. Préparer les données
X = df[FEATURES].copy()
y = df[TARGET].copy()

# Gérer les valeurs manquantes
print(f"\\n🔍 Vérification des données...")
missing_x = X.isnull().sum().sum()
missing_y = y.isnull().sum()

if missing_x > 0:
    print(f"⚠️  {{missing_x}} valeurs manquantes dans X → remplissage médiane")
    X = X.fillna(X.median())

if missing_y > 0:
    print(f"⚠️  {{missing_y}} valeurs manquantes dans y → suppression")
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

print(f"✅ Données nettoyées: {{len(X):,}} lignes")

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\\n📊 Split train/test:")
print(f"   Train: {{X_train.shape[0]:,}} lignes")
print(f"   Test:  {{X_test.shape[0]:,}} lignes")

# 5. Entraîner les modèles
print(f"\\n🏋️ Entraînement des modèles...")
print("="*70)

models_trained = {{}}

# Linear Regression
try:
    print("\\n1️⃣ Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    score_train = lr.score(X_train, y_train)
    score_test = lr.score(X_test, y_test)
    print(f"   ✅ Entraîné → Train R²: {{score_train:.4f}}, Test R²: {{score_test:.4f}}")
    joblib.dump(lr, 'models/linear_regression_final.pkl')
    models_trained['Linear Regression'] = lr
except Exception as e:
    print(f"   ❌ Erreur: {{str(e)[:100]}}")

# Random Forest
try:
    print("\\n2️⃣ Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train, y_train)
    score_train = rf.score(X_train, y_train)
    score_test = rf.score(X_test, y_test)
    print(f"   ✅ Entraîné → Train R²: {{score_train:.4f}}, Test R²: {{score_test:.4f}}")
    joblib.dump(rf, 'models/random_forest_champion_model.pkl')
    models_trained['Random Forest'] = rf
except Exception as e:
    print(f"   ❌ Erreur: {{str(e)[:100]}}")

# Gradient Boosting
try:
    print("\\n3️⃣ Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=0
    )
    gb.fit(X_train, y_train)
    score_train = gb.score(X_train, y_train)
    score_test = gb.score(X_test, y_test)
    print(f"   ✅ Entraîné → Train R²: {{score_train:.4f}}, Test R²: {{score_test:.4f}}")
    joblib.dump(gb, 'models/gradient_boosting_champion_model.pkl')
    models_trained['Gradient Boosting'] = gb
except Exception as e:
    print(f"   ❌ Erreur: {{str(e)[:100]}}")

# LightGBM
try:
    from lightgbm import LGBMRegressor
    print("\\n4️⃣ LightGBM...")
    lgbm = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=-1,
        force_col_wise=True
    )
    lgbm.fit(X_train, y_train)
    score_train = lgbm.score(X_train, y_train)
    score_test = lgbm.score(X_test, y_test)
    print(f"   ✅ Entraîné → Train R²: {{score_train:.4f}}, Test R²: {{score_test:.4f}}")
    joblib.dump(lgbm, 'models/lightgbm_robust_model.pkl')
    models_trained['LightGBM'] = lgbm
except ImportError:
    print("\\n4️⃣ LightGBM: ⏭️  Non installé (pip install lightgbm)")
except Exception as e:
    print(f"   ❌ Erreur: {{str(e)[:100]}}")

# Voting Regressor
try:
    print("\\n5️⃣ Voting Regressor (Ensemble)...")
    voting = VotingRegressor([
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42))
    ])
    voting.fit(X_train, y_train)
    score_train = voting.score(X_train, y_train)
    score_test = voting.score(X_test, y_test)
    print(f"   ✅ Entraîné → Train R²: {{score_train:.4f}}, Test R²: {{score_test:.4f}}")
    joblib.dump(voting, 'models/VOTING_REGRESSOR_FINAL_CHAMPION.pkl')
    models_trained['Voting Regressor'] = voting
except Exception as e:
    print(f"   ❌ Erreur: {{str(e)[:100]}}")

# 6. Évaluation finale
print("\\n" + "="*70)
print("📊 ÉVALUATION FINALE SUR LE TEST SET")
print("="*70)

results = []

for name, model in models_trained.items():
    try:
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results.append({{
            'Modèle': name,
            'R² Score': r2,
            'RMSE': rmse,
            'MAE': mae
        }})
        
        print(f"\\n{{name:25}}")
        print(f"   R² Score: {{r2:.4f}}")
        print(f"   RMSE:     {{rmse:.2f}}")
        print(f"   MAE:      {{mae:.2f}}")
        
    except Exception as e:
        print(f"\\n{{name:25}}")
        print(f"   ❌ Erreur: {{str(e)[:80]}}")

# 7. Résumé et sauvegarde
if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R² Score', ascending=False)
    results_df.to_csv('models/evaluation_results.csv', index=False)
    
    print("\\n" + "="*70)
    print("🏆 CLASSEMENT DES MODÈLES")
    print("="*70)
    print(results_df.to_string(index=False))
    
    print("\\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ!")
    print("="*70)
    print(f"📦 {{len(models_trained)}} modèles sauvegardés dans models/")
    print(f"🏆 Champion: {{results_df.iloc[0]['Modèle']}} (R² = {{results_df.iloc[0]['R² Score']:.4f}})")
    print("📊 Résultats sauvegardés: models/evaluation_results.csv")
else:
    print("\\n❌ Aucun modèle n'a pu être entraîné")
'''
    
    return code


if __name__ == "__main__":
    print("="*70)
    print("🔧 CORRECTION CSV - GLOBAL SUPERSTORE")
    print("="*70)
    
    csv_path = "data/Global_Superstore_100%_PROPRE_51290.csv"
    output_path = "data/Global_Superstore_FIXED.csv"
    
    # Vérifier l'existence
    if not Path(csv_path).exists():
        print(f"\\n❌ Fichier non trouvé: {csv_path}")
        exit(1)
    
    # Corriger le CSV
    try:
        df = fix_csv_advanced(csv_path, output_path)
        
        print("\\n" + "="*70)
        print("🎉 CSV CORRIGÉ AVEC SUCCÈS!")
        print("="*70)
        
        # Vérifier qu'on a des colonnes numériques
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) == 0:
            print("\\n⚠️  ATTENTION: Aucune colonne numérique détectée!")
            print("Les modèles ML nécessitent des colonnes numériques.")
        else:
            # Générer le script d'entraînement
            print("\\n🔧 Génération du script d'entraînement...")
            training_code = create_training_script_v2(df)
            
            with open("train_models.py", "w", encoding='utf-8') as f:
                f.write(training_code)
            
            print("✅ Script généré: train_models.py")
            
            print(f"""
\\n📋 PROCHAINES ÉTAPES:

1️⃣ Utilisez le CSV corrigé dans votre app.py:
   df = pd.read_csv("data/Global_Superstore_FIXED.csv")

2️⃣ Entraînez vos modèles:
   python train_models.py

3️⃣ Lancez Streamlit:
   streamlit run app.py

✅ Tout est prêt! {{len(numeric_cols)}} colonnes numériques disponibles
            """)
        
    except Exception as e:
        print(f"\\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print("""
\\n💡 SOLUTION ALTERNATIVE:

Le CSV a une structure complexe avec des guillemets imbriqués.

Option 1 - Excel:
1. Ouvrez le fichier dans Excel
2. Fichier → Enregistrer sous → CSV UTF-8
3. Fermez et rouvrez pour vérifier

Option 2 - Python manuel:
1. Ouvrez le fichier dans un éditeur de texte
2. Remplacez tous les "" par " (guillemets doubles → simples)
3. Sauvegardez

Option 3 - LibreOffice:
1. Ouvrez dans LibreOffice Calc
2. Choisissez le séparateur virgule et guillemets comme délimiteur de texte
3. Exportez en CSV standard
        """)
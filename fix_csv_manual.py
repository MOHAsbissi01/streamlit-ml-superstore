"""
Parser manuel FINAL pour le CSV problématique
"""

import pandas as pd
import re
from pathlib import Path

def parse_csv_line(line):
    """
    Parse une ligne CSV avec guillemets complexes
    """
    # Retirer le guillemet de début et de fin
    line = line.strip()
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]
    
    # Parser manuellement en gérant les guillemets doubles
    values = []
    current_value = ""
    in_quotes = False
    i = 0
    
    while i < len(line):
        char = line[i]
        
        if char == '"':
            # Vérifier si c'est un guillemet double ""
            if i + 1 < len(line) and line[i + 1] == '"':
                current_value += '"'
                i += 2
                continue
            else:
                # Toggle l'état des guillemets
                in_quotes = not in_quotes
                i += 1
                continue
        
        if char == ',' and not in_quotes:
            # C'est un séparateur
            values.append(current_value)
            current_value = ""
            i += 1
            continue
        
        current_value += char
        i += 1
    
    # Ajouter la dernière valeur
    if current_value or line.endswith(','):
        values.append(current_value)
    
    return values


def fix_csv_final(input_path, output_path):
    """
    Parse le CSV avec gestion des colonnes variables
    """
    print("="*70)
    print("🔧 PARSING FINAL DU CSV")
    print("="*70)
    
    print(f"\n📥 Lecture de: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"✅ {len(lines)} lignes lues")
    
    # Parser le header
    print("\n🔍 Parsing du header...")
    header = parse_csv_line(lines[0])
    n_cols = len(header)
    
    print(f"✅ Header: {n_cols} colonnes")
    print(f"   Colonnes: {', '.join(header)}")
    
    # Analyser les lignes pour trouver le nombre de colonnes le plus fréquent
    print(f"\n🔍 Analyse de la structure...")
    col_counts = {}
    sample_size = min(1000, len(lines) - 1)
    
    for line in lines[1:sample_size+1]:
        values = parse_csv_line(line)
        n = len(values)
        col_counts[n] = col_counts.get(n, 0) + 1
    
    print(f"   Distribution des colonnes (sur {sample_size} lignes):")
    for n, count in sorted(col_counts.items()):
        print(f"   - {n} colonnes: {count} lignes ({count/sample_size*100:.1f}%)")
    
    # Utiliser le nombre de colonnes le plus fréquent
    most_common_cols = max(col_counts.items(), key=lambda x: x[1])[0]
    
    if most_common_cols != n_cols:
        print(f"\n⚠️  Le header a {n_cols} colonnes, mais la majorité des lignes en ont {most_common_cols}")
        
        if most_common_cols > n_cols:
            # Ajouter des colonnes supplémentaires
            for i in range(n_cols, most_common_cols):
                header.append(f"Extra_Column_{i-n_cols+1}")
            print(f"   → Ajout de {most_common_cols - n_cols} colonnes supplémentaires")
        
        n_cols = most_common_cols
    
    # Parser toutes les lignes avec le bon nombre de colonnes
    print(f"\n🔄 Parsing avec {n_cols} colonnes attendues...")
    data_rows = []
    skipped = 0
    
    for i, line in enumerate(lines[1:], start=1):
        if i % 10000 == 0:
            print(f"   ⏳ {i:,} lignes parsées... ({len(data_rows):,} valides)")
        
        try:
            values = parse_csv_line(line)
            
            if len(values) == n_cols:
                data_rows.append(values)
            elif len(values) < n_cols:
                # Compléter avec des valeurs vides
                values.extend([''] * (n_cols - len(values)))
                data_rows.append(values)
            else:
                # Trop de colonnes - probablement une virgule dans un champ
                # Garder les n_cols premières colonnes
                data_rows.append(values[:n_cols])
                
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"   ❌ Ligne {i}: {str(e)[:50]}")
    
    print(f"\n✅ Parsing terminé:")
    print(f"   - Lignes valides: {len(data_rows):,}")
    print(f"   - Lignes problématiques: {skipped:,}")
    
    # Créer le DataFrame
    print(f"\n📊 Création du DataFrame...")
    df = pd.DataFrame(data_rows, columns=header)
    
    print(f"✅ DataFrame créé: {df.shape}")
    
    # NE PAS tout convertir en numérique - garder les types appropriés
    print(f"\n🔄 Détection des types de colonnes...")
    
    numeric_cols = []
    text_cols = []
    
    for col in df.columns:
        # Essayer de convertir en numérique
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
            
            # Si plus de 80% des valeurs sont converties, c'est numérique
            if converted.notna().sum() / len(df) > 0.8:
                df[col] = converted
                numeric_cols.append(col)
            else:
                # Garder comme texte
                text_cols.append(col)
        except:
            text_cols.append(col)
    
    print(f"✅ Types détectés:")
    print(f"   - Colonnes numériques: {len(numeric_cols)}")
    print(f"   - Colonnes textuelles: {len(text_cols)}")
    
    if numeric_cols:
        print(f"\n🔢 Colonnes numériques:")
        for col in numeric_cols:
            non_null = df[col].notna().sum()
            print(f"   • {col:30} ({non_null:,} valeurs)")
    
    if text_cols:
        print(f"\n📝 Colonnes textuelles:")
        for col in text_cols[:10]:  # Afficher les 10 premières
            unique = df[col].nunique()
            print(f"   • {col:30} ({unique:,} valeurs uniques)")
    
    # Sauvegarder
    if output_path:
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n💾 Sauvegardé: {output_path}")
    
    # Aperçu
    print(f"\n📊 Aperçu des premières lignes:")
    print(df.head())
    
    print(f"\n📊 Info complète:")
    print(df.info())
    
    return df


def create_optimized_training_script(df):
    """
    Crée un script d'entraînement avec les bonnes colonnes
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Trouver la target
    target_candidates = ['Profit', 'Sales', 'Revenue']
    target = None
    
    for candidate in target_candidates:
        if candidate in numeric_cols:
            target = candidate
            break
    
    if target is None and numeric_cols:
        target = numeric_cols[0]
    
    features = [col for col in numeric_cols if col != target]
    
    code = f'''"""
Script d'entraînement - Global Superstore
Généré automatiquement
"""

import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

os.makedirs("models", exist_ok=True)

print("="*70)
print("🚀 ENTRAÎNEMENT - GLOBAL SUPERSTORE")
print("="*70)

# Charger les données
print("\\n📥 Chargement...")
df = pd.read_csv("data/Global_Superstore_FIXED.csv")
print(f"✅ {{df.shape[0]:,}} lignes × {{df.shape[1]}} colonnes")

# Configuration
TARGET = '{target}'
FEATURES = {features}

print(f"\\n🎯 Configuration:")
print(f"   Target: {{TARGET}}")
print(f"   Features: {{len(FEATURES)}}")

# Préparer les données
X = df[FEATURES].copy()
y = df[TARGET].copy()

# Nettoyer
print(f"\\n🔍 Nettoyage...")
if X.isnull().sum().sum() > 0:
    print(f"   Remplissage de {{X.isnull().sum().sum()}} valeurs manquantes")
    X = X.fillna(X.median())

if y.isnull().sum() > 0:
    print(f"   Suppression de {{y.isnull().sum()}} lignes avec target manquante")
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

print(f"✅ {{len(X):,}} lignes valides")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\\n📊 Split: {{X_train.shape[0]:,}} train / {{X_test.shape[0]:,}} test")

# Entraînement
print(f"\\n🏋️ Entraînement...")
models = {{}}

# Linear Regression
try:
    print("\\n1️⃣ Linear Regression")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print(f"   Train R²: {{lr.score(X_train, y_train):.4f}}")
    print(f"   Test R²:  {{lr.score(X_test, y_test):.4f}}")
    joblib.dump(lr, 'models/linear_regression_final.pkl')
    models['Linear Regression'] = lr
except Exception as e:
    print(f"   ❌ {{str(e)[:80]}}")

# Random Forest
try:
    print("\\n2️⃣ Random Forest")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"   Train R²: {{rf.score(X_train, y_train):.4f}}")
    print(f"   Test R²:  {{rf.score(X_test, y_test):.4f}}")
    joblib.dump(rf, 'models/random_forest_champion_model.pkl')
    models['Random Forest'] = rf
except Exception as e:
    print(f"   ❌ {{str(e)[:80]}}")

# Gradient Boosting
try:
    print("\\n3️⃣ Gradient Boosting")
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    print(f"   Train R²: {{gb.score(X_train, y_train):.4f}}")
    print(f"   Test R²:  {{gb.score(X_test, y_test):.4f}}")
    joblib.dump(gb, 'models/gradient_boosting_champion_model.pkl')
    models['Gradient Boosting'] = gb
except Exception as e:
    print(f"   ❌ {{str(e)[:80]}}")

# Voting Regressor
try:
    print("\\n4️⃣ Voting Regressor")
    voting = VotingRegressor([
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42))
    ])
    voting.fit(X_train, y_train)
    print(f"   Train R²: {{voting.score(X_train, y_train):.4f}}")
    print(f"   Test R²:  {{voting.score(X_test, y_test):.4f}}")
    joblib.dump(voting, 'models/VOTING_REGRESSOR_FINAL_CHAMPION.pkl')
    models['Voting Regressor'] = voting
except Exception as e:
    print(f"   ❌ {{str(e)[:80]}}")

# Évaluation
print(f"\\n" + "="*70)
print("📊 ÉVALUATION FINALE")
print("="*70)

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results.append({{'Modèle': name, 'R²': r2, 'RMSE': rmse, 'MAE': mae}})
    print(f"\\n{{name:25}} R²: {{r2:.4f}}  RMSE: {{rmse:.2f}}  MAE: {{mae:.2f}}")

results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
results_df.to_csv('models/evaluation_results.csv', index=False)

print(f"\\n" + "="*70)
print("✅ TERMINÉ!")
print(f"🏆 Champion: {{results_df.iloc[0]['Modèle']}} (R² = {{results_df.iloc[0]['R²']:.4f}})")
print(f"📦 {{len(models)}} modèles dans models/")
print("="*70)
'''
    
    return code


if __name__ == "__main__":
    csv_path = "data/Global_Superstore_100%_PROPRE_51290.csv"
    output_path = "data/Global_Superstore_FIXED.csv"
    
    if not Path(csv_path).exists():
        print(f"❌ Fichier non trouvé: {csv_path}")
        exit(1)
    
    try:
        df = fix_csv_final(csv_path, output_path)
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 5:
            print("\n" + "="*70)
            print("🎉 SUCCÈS COMPLET!")
            print("="*70)
            
            # Générer le script d'entraînement
            training_code = create_optimized_training_script(df)
            
            with open("train_models.py", "w", encoding='utf-8') as f:
                f.write(training_code)
            
            print(f"""
✅ CSV corrigé: {{df.shape[0]:,}} lignes × {{df.shape[1]}} colonnes
✅ {{len(numeric_cols)}} colonnes numériques
✅ Script généré: train_models.py

📋 PROCHAINES ÉTAPES:

1️⃣ Le CSV est prêt: data/Global_Superstore_FIXED.csv

2️⃣ Entraînez les modèles:
   python train_models.py

3️⃣ Lancez Streamlit:
   streamlit run app.py

🎯 Colonnes numériques: {{', '.join(numeric_cols[:5])}}...
            """)
        else:
            print(f"\\n⚠️  Seulement {{len(numeric_cols)}} colonnes numériques")
            print("Les modèles ML nécessitent plus de features numériques.")
    
    except Exception as e:
        print(f"\\n❌ ERREUR: {{e}}")
        import traceback
        traceback.print_exc()
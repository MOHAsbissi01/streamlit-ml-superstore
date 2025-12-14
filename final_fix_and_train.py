"""
Fix final du CSV + Entraînement des modèles
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("="*70)
print("🔧 FIX FINAL + ENTRAÎNEMENT DES MODÈLES")
print("="*70)

# 1. Charger et corriger le CSV
print("\n📥 Chargement du CSV...")
df = pd.read_csv("data/Global_Superstore_FIXED.csv")
print(f"✅ Chargé: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# 2. Convertir Sales en numérique (c'était resté en texte)
print("\n🔄 Correction de la colonne Sales...")
if df['Sales'].dtype == 'object':
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    print(f"✅ Sales converti en numérique: {df['Sales'].notna().sum():,} valeurs")

# Convertir Shipping_Delay_Days aussi
if 'Shipping_Delay_Days' in df.columns and df['Shipping_Delay_Days'].dtype == 'object':
    df['Shipping_Delay_Days'] = pd.to_numeric(df['Shipping_Delay_Days'], errors='coerce')
    print(f"✅ Shipping_Delay_Days converti: {df['Shipping_Delay_Days'].notna().sum():,} valeurs")

# 3. Sauvegarder la version corrigée
df.to_csv("data/Global_Superstore_FIXED.csv", index=False)
print("💾 CSV corrigé sauvegardé")

# 4. Afficher les colonnes numériques
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
text_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n📊 Colonnes finales:")
print(f"   - Numériques ({len(numeric_cols)}): {', '.join(numeric_cols)}")
print(f"   - Textuelles ({len(text_cols)}): {', '.join(text_cols[:5])}...")

# 5. Préparation pour ML
print("\n" + "="*70)
print("🚀 ENTRAÎNEMENT DES MODÈLES")
print("="*70)

os.makedirs("models", exist_ok=True)

# Choisir la target et les features
TARGET = 'Profit'  # Variable à prédire

# Retirer Extra_Column_1 qui a beaucoup de NaN
features_to_exclude = [TARGET, 'Extra_Column_1']
FEATURES = [col for col in numeric_cols if col not in features_to_exclude]

print(f"\n🎯 Configuration:")
print(f"   Target: {TARGET}")
print(f"   Features ({len(FEATURES)}): {', '.join(FEATURES)}")

# Préparer X et y
X = df[FEATURES].copy()
y = df[TARGET].copy()

# Nettoyer les NaN
print(f"\n🔍 Nettoyage des données...")
print(f"   NaN dans X: {X.isnull().sum().sum()}")
print(f"   NaN dans y: {y.isnull().sum()}")

# Remplir les NaN dans X avec la médiane
if X.isnull().sum().sum() > 0:
    X = X.fillna(X.median())
    print(f"✅ NaN dans X remplis avec la médiane")

# Retirer les lignes où y est NaN
if y.isnull().sum() > 0:
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    print(f"✅ {len(X):,} lignes valides conservées")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Split:")
print(f"   Train: {X_train.shape[0]:,} lignes")
print(f"   Test:  {X_test.shape[0]:,} lignes")

# 6. Entraîner les modèles
print(f"\n" + "="*70)
print("🏋️ ENTRAÎNEMENT EN COURS...")
print("="*70)

models_trained = {}
training_results = []

# Linear Regression
print("\n1️⃣ Linear Regression...")
try:
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    train_score = lr.score(X_train, y_train)
    test_score = lr.score(X_test, y_test)
    print(f"   ✅ Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    joblib.dump(lr, 'models/linear_regression_final.pkl')
    models_trained['Linear Regression'] = lr
    training_results.append({'model': 'Linear Regression', 'train_r2': train_score, 'test_r2': test_score})
except Exception as e:
    print(f"   ❌ Erreur: {str(e)[:100]}")

# Random Forest
print("\n2️⃣ Random Forest...")
try:
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train, y_train)
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    print(f"   ✅ Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    joblib.dump(rf, 'models/random_forest_champion_model.pkl')
    models_trained['Random Forest'] = rf
    training_results.append({'model': 'Random Forest', 'train_r2': train_score, 'test_r2': test_score})
except Exception as e:
    print(f"   ❌ Erreur: {str(e)[:100]}")

# Gradient Boosting
print("\n3️⃣ Gradient Boosting...")
try:
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=0
    )
    gb.fit(X_train, y_train)
    train_score = gb.score(X_train, y_train)
    test_score = gb.score(X_test, y_test)
    print(f"   ✅ Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    joblib.dump(gb, 'models/gradient_boosting_champion_model.pkl')
    models_trained['Gradient Boosting'] = gb
    training_results.append({'model': 'Gradient Boosting', 'train_r2': train_score, 'test_r2': test_score})
except Exception as e:
    print(f"   ❌ Erreur: {str(e)[:100]}")

# LightGBM (optionnel)
print("\n4️⃣ LightGBM...")
try:
    from lightgbm import LGBMRegressor
    lgbm = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=-1,
        force_col_wise=True
    )
    lgbm.fit(X_train, y_train)
    train_score = lgbm.score(X_train, y_train)
    test_score = lgbm.score(X_test, y_test)
    print(f"   ✅ Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    joblib.dump(lgbm, 'models/lightgbm_robust_model.pkl')
    models_trained['LightGBM'] = lgbm
    training_results.append({'model': 'LightGBM', 'train_r2': train_score, 'test_r2': test_score})
except ImportError:
    print("   ⏭️  Non installé (pip install lightgbm)")
except Exception as e:
    print(f"   ❌ Erreur: {str(e)[:100]}")

# Voting Regressor (Ensemble)
print("\n5️⃣ Voting Regressor (Champion)...")
try:
    voting = VotingRegressor([
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42))
    ])
    voting.fit(X_train, y_train)
    train_score = voting.score(X_train, y_train)
    test_score = voting.score(X_test, y_test)
    print(f"   ✅ Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    joblib.dump(voting, 'models/VOTING_REGRESSOR_FINAL_CHAMPION.pkl')
    models_trained['Voting Regressor'] = voting
    training_results.append({'model': 'Voting Regressor', 'train_r2': train_score, 'test_r2': test_score})
except Exception as e:
    print(f"   ❌ Erreur: {str(e)[:100]}")

# 7. Évaluation finale
print(f"\n" + "="*70)
print("📊 ÉVALUATION FINALE SUR LE TEST SET")
print("="*70)

results = []

for name, model in models_trained.items():
    try:
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results.append({
            'Modèle': name,
            'R² Score': r2,
            'RMSE': rmse,
            'MAE': mae
        })
        
        print(f"\n{name:25}")
        print(f"   R² Score: {r2:.4f}")
        print(f"   RMSE:     {rmse:,.2f}")
        print(f"   MAE:      {mae:,.2f}")
        
    except Exception as e:
        print(f"\n{name:25}")
        print(f"   ❌ Erreur: {str(e)[:80]}")

# Sauvegarder les résultats
if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R² Score', ascending=False)
    results_df.to_csv('models/evaluation_results.csv', index=False)
    
    print("\n" + "="*70)
    print("🏆 CLASSEMENT DES MODÈLES")
    print("="*70)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*70)
    print(f"📦 {len(models_trained)} modèles sauvegardés dans models/")
    print(f"🏆 Champion: {results_df.iloc[0]['Modèle']} (R² = {results_df.iloc[0]['R² Score']:.4f})")
    print(f"📊 Résultats: models/evaluation_results.csv")
    
    print(f"""
\n📋 PROCHAINE ÉTAPE:

Lancez votre application Streamlit:
   streamlit run app.py

Tous les modèles sont prêts et fonctionnels! 🚀
    """)
else:
    print("\n❌ Aucun modèle n'a pu être entraîné")
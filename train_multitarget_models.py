# -*- coding: utf-8 -*-
"""
Script d'entrainement complet pour plusieurs targets
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path

# Force UTF-8 encoding for output
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("="*80)
print("ENTRAINEMENT MULTI-TARGET - GLOBAL SUPERSTORE")
print("="*80)

# Configuration
TARGETS = ['Sales', 'Profit', 'Quantity']
os.makedirs("models", exist_ok=True)

# 1. Charger et pretraiter les donnees
print("\nChargement des donnees...")
from src.preprocessor import load_and_preprocess_data

df = load_and_preprocess_data("data/Global_Superstore_FIXED.csv")
print(f"Donnees pretraitees: {df.shape[0]:,} lignes x {df.shape[1]} colonnes")

# Colonnes numeriques disponibles
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nColonnes numeriques: {len(numeric_cols)}")
print(f"   {', '.join(numeric_cols)}")

# Dictionnaire pour stocker tous les resultats
all_results = []

# 2. Boucle sur chaque target
for target in TARGETS:
    print(f"\n{'='*80}")
    print(f"TARGET: {target}")
    print(f"{'='*80}")
    
    if target not in df.columns:
        print(f"WARNING: {target} non trouve, skip")
        continue
    
    # Features = toutes les colonnes sauf la target et Extra_Column_1
    features_to_exclude = [target, 'Extra_Column_1']
    features = [col for col in numeric_cols if col not in features_to_exclude]
    
    print(f"\nConfiguration:")
    print(f"   Target: {target}")
    print(f"   Features: {len(features)} colonnes")
    
    # Preparer X et y
    X = df[features].copy()
    y = df[target].copy()
    
    # Nettoyer les NaN
    print(f"\nNettoyage...")
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())
        print(f"   NaN dans X remplis")
    
    if y.isnull().sum() > 0:
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        print(f"   {len(X):,} lignes valides")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nSplit: Train={X_train.shape[0]:,}, Test={X_test.shape[0]:,}")
    
    # 3. Entrainer les modeles
    print(f"\nEntrainement des modeles...")
    print("-"*80)
    
    # Linear Regression
    print(f"\n[1/5] Linear Regression...")
    try:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        train_r2 = lr.score(X_train, y_train)
        test_r2 = lr.score(X_test, y_test)
        y_pred = lr.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"   Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}, RMSE: {rmse:.2f}")
        
        model_filename = f'models/linear_regression_{target.lower()}.pkl'
        joblib.dump(lr, model_filename)
        print(f"   Saved: {model_filename}")
        
        all_results.append({
            'Target': target,
            'Model': 'Linear Regression',
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'RMSE': rmse,
            'Filename': model_filename
        })
    except Exception as e:
        print(f"   ERROR: {str(e)[:100]}")
    
    # Random Forest
    print(f"\n[2/5] Random Forest...")
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
        train_r2 = rf.score(X_train, y_train)
        test_r2 = rf.score(X_test, y_test)
        y_pred = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"   Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}, RMSE: {rmse:.2f}")
        
        model_filename = f'models/random_forest_{target.lower()}.pkl'
        joblib.dump(rf, model_filename)
        print(f"   Saved: {model_filename}")
        
        all_results.append({
            'Target': target,
            'Model': 'Random Forest',
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'RMSE': rmse,
            'Filename': model_filename
        })
    except Exception as e:
        print(f"   ERROR: {str(e)[:100]}")
    
    # Gradient Boosting
    print(f"\n[3/5] Gradient Boosting...")
    try:
        gb = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=0
        )
        gb.fit(X_train, y_train)
        train_r2 = gb.score(X_train, y_train)
        test_r2 = gb.score(X_test, y_test)
        y_pred = gb.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"   Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}, RMSE: {rmse:.2f}")
        
        model_filename = f'models/gradient_boosting_{target.lower()}.pkl'
        joblib.dump(gb, model_filename)
        print(f"   Saved: {model_filename}")
        
        all_results.append({
            'Target': target,
            'Model': 'Gradient Boosting',
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'RMSE': rmse,
            'Filename': model_filename
        })
    except Exception as e:
        print(f"   ERROR: {str(e)[:100]}")
    
    # LightGBM
    print(f"\n[4/5] LightGBM...")
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
        train_r2 = lgbm.score(X_train, y_train)
        test_r2 = lgbm.score(X_test, y_test)
        y_pred = lgbm.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"   Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}, RMSE: {rmse:.2f}")
        
        model_filename = f'models/lightgbm_{target.lower()}.pkl'
        joblib.dump(lgbm, model_filename)
        print(f"   Saved: {model_filename}")
        
        all_results.append({
            'Target': target,
            'Model': 'LightGBM',
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'RMSE': rmse,
            'Filename': model_filename
        })
    except ImportError:
        print("   SKIPPED: LightGBM not installed")
    except Exception as e:
        print(f"   ERROR: {str(e)[:100]}")
    
    # Voting Regressor
    print(f"\n[5/5] Voting Regressor...")
    try:
        voting = VotingRegressor([
            ('lr', LinearRegression()),
            ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42))
        ])
        voting.fit(X_train, y_train)
        train_r2 = voting.score(X_train, y_train)
        test_r2 = voting.score(X_test, y_test)
        y_pred = voting.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"   Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}, RMSE: {rmse:.2f}")
        
        model_filename = f'models/voting_regressor_{target.lower()}.pkl'
        joblib.dump(voting, model_filename)
        print(f"   Saved: {model_filename}")
        
        all_results.append({
            'Target': target,
            'Model': 'Voting Regressor',
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'RMSE': rmse,
            'Filename': model_filename
        })
    except Exception as e:
        print(f"   ERROR: {str(e)[:100]}")

# 4. Resume final
print(f"\n{'='*80}")
print("RESUME FINAL")
print(f"{'='*80}")

if all_results:
    results_df = pd.DataFrame(all_results)
    
    # Sauvegarder les resultats
    results_df.to_csv('models/training_results_multitarget.csv', index=False)
    
    # Afficher par target
    for target in TARGETS:
        target_results = results_df[results_df['Target'] == target]
        if not target_results.empty:
            print(f"\n{target}:")
            print(target_results[['Model', 'Test_R2', 'RMSE']].to_string(index=False))
            best = target_results.loc[target_results['Test_R2'].idxmax()]
            print(f"   BEST: {best['Model']} (R2={best['Test_R2']:.4f})")
    
    print(f"\n{'='*80}")
    print(f"ENTRAINEMENT TERMINE!")
    print(f"{'='*80}")
    print(f"{len(all_results)} modeles sauvegardes dans models/")
    print(f"Resultats: models/training_results_multitarget.csv")
    print(f"\nProchaine etape: Mettre a jour le model_loader.py et l'app.py")
else:
    print("\nERROR: Aucun modele n'a pu etre entraine")

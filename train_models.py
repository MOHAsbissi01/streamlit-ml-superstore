"""
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
print("\n📥 Chargement...")
df = pd.read_csv("data/Global_Superstore_FIXED.csv")
print(f"✅ {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# Configuration
TARGET = 'Profit'
FEATURES = ['Quantity', 'Discount', 'Shipping Cost', 'Order Priority', 'Order_Year', 'Order_Month', 'Order_DayOfWeek', 'Order_Quarter', 'is_big_sale', 'is_big_shipping', 'is_big_loss', 'is_high_discount', 'Extra_Column_1']

print(f"\n🎯 Configuration:")
print(f"   Target: {TARGET}")
print(f"   Features: {len(FEATURES)}")

# Préparer les données
X = df[FEATURES].copy()
y = df[TARGET].copy()

# Nettoyer
print(f"\n🔍 Nettoyage...")
if X.isnull().sum().sum() > 0:
    print(f"   Remplissage de {X.isnull().sum().sum()} valeurs manquantes")
    X = X.fillna(X.median())

if y.isnull().sum() > 0:
    print(f"   Suppression de {y.isnull().sum()} lignes avec target manquante")
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

print(f"✅ {len(X):,} lignes valides")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Split: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test")

# Entraînement
print(f"\n🏋️ Entraînement...")
models = {}

# Linear Regression
try:
    print("\n1️⃣ Linear Regression")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print(f"   Train R²: {lr.score(X_train, y_train):.4f}")
    print(f"   Test R²:  {lr.score(X_test, y_test):.4f}")
    joblib.dump(lr, 'models/linear_regression_final.pkl')
    models['Linear Regression'] = lr
except Exception as e:
    print(f"   ❌ {str(e)[:80]}")

# Random Forest
try:
    print("\n2️⃣ Random Forest")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"   Train R²: {rf.score(X_train, y_train):.4f}")
    print(f"   Test R²:  {rf.score(X_test, y_test):.4f}")
    joblib.dump(rf, 'models/random_forest_champion_model.pkl')
    models['Random Forest'] = rf
except Exception as e:
    print(f"   ❌ {str(e)[:80]}")

# Gradient Boosting
try:
    print("\n3️⃣ Gradient Boosting")
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    print(f"   Train R²: {gb.score(X_train, y_train):.4f}")
    print(f"   Test R²:  {gb.score(X_test, y_test):.4f}")
    joblib.dump(gb, 'models/gradient_boosting_champion_model.pkl')
    models['Gradient Boosting'] = gb
except Exception as e:
    print(f"   ❌ {str(e)[:80]}")

# Voting Regressor
try:
    print("\n4️⃣ Voting Regressor")
    voting = VotingRegressor([
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42))
    ])
    voting.fit(X_train, y_train)
    print(f"   Train R²: {voting.score(X_train, y_train):.4f}")
    print(f"   Test R²:  {voting.score(X_test, y_test):.4f}")
    joblib.dump(voting, 'models/VOTING_REGRESSOR_FINAL_CHAMPION.pkl')
    models['Voting Regressor'] = voting
except Exception as e:
    print(f"   ❌ {str(e)[:80]}")

# Évaluation
print(f"\n" + "="*70)
print("📊 ÉVALUATION FINALE")
print("="*70)

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results.append({'Modèle': name, 'R²': r2, 'RMSE': rmse, 'MAE': mae})
    print(f"\n{name:25} R²: {r2:.4f}  RMSE: {rmse:.2f}  MAE: {mae:.2f}")

results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
results_df.to_csv('models/evaluation_results.csv', index=False)

print(f"\n" + "="*70)
print("✅ TERMINÉ!")
print(f"🏆 Champion: {results_df.iloc[0]['Modèle']} (R² = {results_df.iloc[0]['R²']:.4f})")
print(f"📦 {len(models)} modèles dans models/")
print("="*70)

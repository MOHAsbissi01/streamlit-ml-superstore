"""
Script pour vérifier les colonnes du CSV
"""

import pandas as pd

# Charger le CSV
df = pd.read_csv("data/Global_Superstore_FIXED.csv")

print("="*60)
print("🔍 ANALYSE DU CSV")
print("="*60)

print(f"\n📊 Dimensions: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

print("\n📋 TOUTES LES COLONNES:")
print("="*60)
for i, col in enumerate(df.columns, 1):
    col_type = df[col].dtype
    n_unique = df[col].nunique()
    print(f"{i:2d}. {col:30s} | Type: {str(col_type):10s} | Valeurs uniques: {n_unique:,}")

print("\n" + "="*60)
print("🔢 COLONNES PAR TYPE:")
print("="*60)

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
text_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n✅ Colonnes numériques ({len(numeric_cols)}):")
for col in numeric_cols:
    print(f"   - {col}")

print(f"\n📝 Colonnes textuelles ({len(text_cols)}):")
for col in text_cols:
    n_unique = df[col].nunique()
    print(f"   - {col:30s} → {n_unique:,} valeurs uniques")

print("\n" + "="*60)
print("⚠️  COLONNES À HAUTE CARDINALITÉ (>100 valeurs uniques):")
print("="*60)

for col in text_cols:
    n_unique = df[col].nunique()
    if n_unique > 100:
        print(f"   ❌ {col:30s} → {n_unique:,} valeurs ← À SUPPRIMER!")

print("\n" + "="*60)
print("✅ COLONNES CATÉGORIELLES VALIDES (<100 valeurs):")
print("="*60)

for col in text_cols:
    n_unique = df[col].nunique()
    if n_unique <= 100:
        print(f"   ✅ {col:30s} → {n_unique} valeurs ← OK pour encodage")
        # Afficher les valeurs
        print(f"      Valeurs: {list(df[col].unique()[:5])}...")
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Charger le CSV
df = pd.read_csv("data/Global_Superstore_100%_PROPRE_51290.csv", quotechar='"')

# Exemple : prédire 'Profit' à partir de colonnes numériques
X = df[['Sales', 'Quantity', 'Discount', 'Shipping Cost']]
y = df['Profit']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définir les modèles
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
voting = VotingRegressor([('lr', lr), ('rf', rf)])

# Entraîner
voting.fit(X_train, y_train)

# Créer le dossier models s'il n'existe pas
import os
os.makedirs("models", exist_ok=True)

# Sauvegarder le modèle
joblib.dump(voting, "models/VOTING_REGRESSOR_FINAL_CHAMPION.pkl")
print("✅ Modèle Voting Regressor entraîné et sauvegardé !")

# 🏪 Global Superstore - ML Model Deployment

## 📝 Description

Application web interactive pour le déploiement et la comparaison de modèles de Machine Learning entraînés sur le dataset Global Superstore.

## ✨ Fonctionnalités

### 🔮 Mode Prédiction Simple
- Sélection de la variable cible
- Choix du modèle
- Saisie interactive des paramètres
- Prédiction en temps réel
- Comparaison avec la distribution du dataset

### 📊 Mode Comparaison de Modèles
- Comparaison de plusieurs modèles simultanément
- Calcul des métriques (R², RMSE, MAE, MSE)
- Visualisations comparatives
- Export des résultats en CSV

### 📈 Mode Analyse du Dataset
- Statistiques descriptives
- Visualisation des distributions
- Analyse exploratoire interactive

## 📁 Structure du Projet

```
projet_ml_comparaison/
│
├── app.py                          # Application Streamlit principale
├── requirements.txt                # Dépendances Python
├── README.md                       # Ce fichier
│
├── data/
│   └── Global_Superstore_100%_PROPRE_51290.csv
│
├── models/                         # Modèles entraînés (.pkl)
│   ├── linear_regression_final.pkl
│   ├── VOTING_REGRESSOR_FINAL_CHAMPION.pkl
│   ├── svr_final.pkl              # À ajouter
│   ├── random_forest_final.pkl    # À ajouter
│   ├── xgboost_final.pkl          # À ajouter
│   └── lightgbm_final.pkl         # À ajouter
│
├── notebooks/
│   └── store.ipynb                # Notebook d'entraînement original
│
└── src/
    ├── __init__.py
    └── model_loader.py            # Module de chargement des modèles
```

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Étapes

1. **Cloner ou télécharger le projet**
```bash
cd projet_ml_comparaison
```

2. **Créer un environnement virtuel (recommandé)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## 💻 Utilisation

### Lancer l'application
```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse: `http://localhost:8501`

### Guide d'utilisation

#### Mode Prédiction Simple 🔮

1. **Sélectionner la variable cible**
   - Choisissez la variable que vous souhaitez prédire

2. **Choisir un modèle**
   - Linear Regression
   - Voting Regressor (Champion)
   - Autres modèles (bientôt disponibles)

3. **Saisir les paramètres**
   - Remplissez les valeurs pour chaque feature
   - Des valeurs par défaut (moyenne) sont proposées

4. **Lancer la prédiction**
   - Cliquez sur "🚀 LANCER LA PRÉDICTION"
   - Visualisez le résultat et sa position dans la distribution

#### Mode Comparaison 📊

1. **Sélectionner la variable cible**

2. **Choisir les modèles à comparer**
   - Sélectionnez plusieurs modèles

3. **Définir le nombre d'échantillons**
   - Slider pour choisir combien de prédictions tester

4. **Lancer la comparaison**
   - Comparez les métriques (R², RMSE, MAE)
   - Visualisez les graphiques comparatifs
   - Téléchargez les résultats

#### Mode Analyse 📈

1. **Explorer les statistiques du dataset**
   - Nombre de lignes, colonnes
   - Valeurs manquantes
   - Types de données

2. **Visualiser les distributions**
   - Histogrammes
   - Box plots
   - Statistiques descriptives

## 🔧 Ajouter de Nouveaux Modèles

### Étape 1: Entraîner et sauvegarder le modèle

Dans ton notebook `store.ipynb`:

```python
import joblib

# Après avoir entraîné ton modèle
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Sauvegarder
joblib.dump(model, 'models/random_forest_final.pkl')
```

### Étape 2: Ajouter dans model_loader.py

```python
def load_models():
    model_files = {
        "Linear Regression": "linear_regression_final.pkl",
        "Voting Regressor (Champion)": "VOTING_REGRESSOR_FINAL_CHAMPION.pkl",
        "Random Forest": "random_forest_final.pkl",  # ← Ajouter ici
        # ... autres modèles
    }
```

### Étape 3: Relancer l'application

```bash
streamlit run app.py
```

Le nouveau modèle apparaîtra automatiquement dans l'interface !

## 📊 Métriques Utilisées

### Régression
- **R² Score**: Coefficient de détermination (0 à 1, plus proche de 1 = meilleur)
- **RMSE**: Root Mean Squared Error (plus petit = meilleur)
- **MAE**: Mean Absolute Error (plus petit = meilleur)
- **MSE**: Mean Squared Error (plus petit = meilleur)

## 🎨 Personnalisation

### Modifier les couleurs

Dans `app.py`, modifiez la section CSS:

```python
st.markdown("""
    <style>
    .prediction-box {
        background: linear-gradient(135deg, #your-color1 0%, #your-color2 100%);
        ...
    }
    </style>
""", unsafe_allow_html=True)
```

### Ajouter de nouvelles visualisations

Dans le mode Comparaison, ajoutez des graphiques Plotly:

```python
import plotly.graph_objects as go

fig = go.Figure(...)
st.plotly_chart(fig, use_container_width=True)
```

## 🐛 Résolution de Problèmes

### Erreur: "Module not found"
```bash
pip install -r requirements.txt --upgrade
```

### Erreur: "Model file not found"
- Vérifiez que les fichiers .pkl sont dans le dossier `models/`
- Vérifiez les noms de fichiers dans `model_loader.py`

### Erreur: "Columns mismatch"
- Assurez-vous que les colonnes du dataset correspondent à celles utilisées lors de l'entraînement
- Vérifiez l'ordre des colonnes

### L'application est lente
- Réduisez le nombre d'échantillons dans le mode Comparaison
- Utilisez `@st.cache_data` et `@st.cache_resource` pour les fonctions coûteuses

## 📈 Performances

| Modèle | R² Score | RMSE | Status |
|--------|----------|------|--------|
| Voting Regressor | 0.XXX | XX.XX | ✅ Disponible |
| Linear Regression | 0.XXX | XX.XX | ✅ Disponible |
| SVR | - | - | ⚠️ Bientôt |
| Random Forest | - | - | ⚠️ Bientôt |
| XGBoost | - | - | ⚠️ Bientôt |
| LightGBM | - | - | ⚠️ Bientôt |

## 🔜 Fonctionnalités à Venir

- [ ] Support complet de tous les modèles
- [ ] Mode batch prediction (upload CSV)
- [ ] Analyse SHAP pour l'interprétabilité
- [ ] API REST pour les prédictions
- [ ] Dashboard de monitoring
- [ ] Export de rapports PDF

## 📚 Ressources

- [Documentation Streamlit](https://docs.streamlit.io)
- [Documentation Scikit-learn](https://scikit-learn.org)
- [Dataset Global Superstore](https://www.kaggle.com/datasets/rohitsahoo/global-superstore)

## 👨‍💻 Développement

### Tester l'application localement

```bash
# Mode debug
streamlit run app.py --logger.level=debug
```

### Tester le chargement des modèles

```bash
# Dans le terminal Python
python src/model_loader.py
```

## 📄 Licence

Ce projet est développé à des fins éducatives.

## 🤝 Contribution

Pour ajouter de nouvelles fonctionnalités:

1. Créer une branche
2. Développer la fonctionnalité
3. Tester localement
4. Documenter les changements

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue.

---

**Bon déploiement! 🚀**
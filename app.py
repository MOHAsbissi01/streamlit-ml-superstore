import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.model_loader import load_models_for_target, get_available_targets
from src.preprocessor import load_and_preprocess_data, prepare_features_for_model

# ----------------------------
# Configuration Streamlit
# ----------------------------
st.set_page_config(
    page_title="ML Model Deployment - Global Superstore",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Titre principal
# ----------------------------
st.markdown('<h1 class="main-header">🏪 Global Superstore - ML Model Deployment</h1>', unsafe_allow_html=True)
st.markdown("### 📊 Déploiement et Comparaison de Modèles de Régression")
st.markdown("---")

# ----------------------------
# Sidebar - Configuration
# ----------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.expander("ℹ️ À propos"):
        st.write("""
        **Projet:** Prédiction avec Global Superstore
        
        **Modèles disponibles:**
        - Linear Regression
        - Random Forest Champion
        - LightGBM variants
        - Voting Regressor
        
        **Dataset:** Global_Superstore_FIXED.csv (prétraité)
        """)
    
    st.markdown("---")
    
    app_mode = st.radio(
        "Mode d'utilisation",
        ["🔮 Prédiction Simple", "📊 Comparaison de Modèles", "📈 Analyse du Dataset"]
    )
    
    st.markdown("---")
    
    with st.expander("⚙️ Paramètres avancés"):
        show_model_details = st.checkbox("Afficher les détails du modèle", value=False)
        show_input_summary = st.checkbox("Afficher le résumé des inputs", value=True)

# ----------------------------
# Chargement du dataset AVEC PRÉTRAITEMENT
# ----------------------------
@st.cache_data
def load_data():
    """Charger et prétraiter le dataset"""
    try:
        df = load_and_preprocess_data("data/Global_Superstore_FIXED.csv")
        st.success(f"✅ Données prétraitées: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
        return df
    except FileNotFoundError:
        st.error("❌ Fichier Global_Superstore_FIXED.csv non trouvé!")
        return None
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        return None

df = load_data()

if df is None:
    st.stop()

# Colonnes numériques
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_columns) == 0:
    st.error("❌ Aucune colonne numérique!")
    st.stop()

# Targets disponibles
available_targets = get_available_targets()

# ----------------------------
# MODE 1: PRÉDICTION SIMPLE
# ----------------------------
if app_mode == "🔮 Prédiction Simple":
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 Configuration")
        
        # Selection du target avec info
        st.info(f"""
        **Targets disponibles avec modèles entraînés:**
        - {', '.join([f'**{t}**' for t in available_targets])}
        """)
        
        target_col = st.selectbox(
            "Variable cible à prédire",
            available_targets,
            help="Sélectionnez le target pour lequel charger les modèles"
        )
        
        # Charger les modèles pour ce target
        @st.cache_resource
        def get_models_for_target(target):
            """Charger les modèles avec cache"""
            try:
                return load_models_for_target(target)
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")
                return {}
        
        models = get_models_for_target(target_col)
        
        if not models:
            st.error(f"❌ Aucun modèle disponible pour {target_col}!")
            st.info("💡 Exécutez `python train_multitarget_models.py` pour entraîner les modèles")
            st.stop()
        
        model_name = st.selectbox(
            "Modèle à utiliser",
            list(models.keys()),
            help="Choisissez le modèle"
        )
        
        model = models[model_name]
        
        if show_model_details:
            st.info(f"""
            **Modèle:** {model_name}
            **Type:** {type(model).__name__}
            **Target:** {target_col}
            """)
            
            if hasattr(model, 'feature_names_in_'):
                st.info(f"**Features attendues:** {len(model.feature_names_in_)}")
    
    with col2:
        st.subheader("📊 Aperçu des Données")
        st.dataframe(df.head(10), width="stretch")
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("📏 Lignes", f"{df.shape[0]:,}")
        with col_stats2:
            st.metric("📊 Colonnes", df.shape[1])
        with col_stats3:
            st.metric("🔢 Numériques", len(numeric_columns))
    
    st.markdown("---")
    
    # Préparer les features
    X, y, missing_features = prepare_features_for_model(df, target_col, model)
    
    if missing_features:
        # Vérifier si le problème est que la target est incompatible
        if target_col in missing_features:
            # Le modèle attend cette colonne comme feature, pas comme target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if hasattr(model, 'feature_names_in_'):
                expected_features = list(model.feature_names_in_)
                possible_targets = [col for col in numeric_cols if col not in expected_features and col != target_col]
            else:
                possible_targets = []
            
            st.error(f"❌ Ce modèle ne peut PAS prédire '{target_col}'")
            st.warning(f"""
            🔍 **Raison**: Le modèle '{model_name}' attend '{target_col}' comme **entrée** (feature), 
            pas comme **sortie** (target à prédire).
            """)
            
            if possible_targets:
                st.info(f"""
                💡 **Solution**: Ce modèle a été entraîné pour prédire l'une de ces variables:
                
                {', '.join([f'**{t}**' for t in possible_targets[:5]])}
                
                ➡️ Changez la "Variable cible à prédire" vers l'une de ces options.
                """)
            else:
                st.info("💡 Essayez un autre modèle ou réentraînez les modèles avec la bonne cible.")
        else:
            # Autres features manquantes
            st.error(f"❌ Ce modèle nécessite {len(missing_features)} features manquantes")
            st.info("💡 Essayez 'Linear Regression' ou 'Random Forest Champion'")
        
        st.stop()
    
    st.success(f"""
    ✅ **Configuration validée:**
    - Variable cible: **{target_col}**
    - Features: **{X.shape[1]}** colonnes
    - Modèle: **{model_name}**
    """)
    
    # Inputs utilisateur
    st.subheader("🔧 Paramètres de Prédiction")
    
    num_cols = 3
    cols = st.columns(num_cols)
    
    input_data = {}
    
    for idx, col in enumerate(X.columns):
        with cols[idx % num_cols]:
            col_min = float(X[col].min())
            col_max = float(X[col].max())
            col_mean = float(X[col].mean())
            
            # Pour les colonnes binaires, utiliser checkbox
            if X[col].nunique() == 2 and set(X[col].unique()).issubset({0, 1, 0.0, 1.0}):
                input_data[col] = float(st.checkbox(
                    label=f"{col}",
                    value=bool(col_mean > 0.5)
                ))
            else:
                input_data[col] = st.number_input(
                    label=f"{col}",
                    min_value=col_min,
                    max_value=col_max,
                    value=col_mean,
                    help=f"Min: {col_min:.2f}, Max: {col_max:.2f}"
                )
    
    input_df = pd.DataFrame([input_data])
    
    if show_input_summary:
        with st.expander("📋 Résumé des valeurs"):
            st.dataframe(input_df.T, width="stretch")
    
    st.markdown("---")
    
    # Prédiction
    col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
    
    with col_pred2:
        if st.button("🚀 LANCER LA PRÉDICTION", key="predict_btn"):
            try:
                with st.spinner("🔄 Calcul en cours..."):
                    prediction = model.predict(input_df)
                    
                st.markdown(f"""
                <div class="prediction-box">
                    🎯 Prédiction de {target_col}: {prediction[0]:.2f}
                </div>
                """, unsafe_allow_html=True)
                
                st.balloons()
                
                st.markdown("### 📊 Contexte de la Prédiction")
                
                col1, col2, col3, col4 = st.columns(4)
                
                actual_mean = df[target_col].mean()
                actual_min = df[target_col].min()
                actual_max = df[target_col].max()
                
                with col1:
                    st.metric("Moyenne Dataset", f"{actual_mean:.2f}")
                with col2:
                    diff_mean = prediction[0] - actual_mean
                    st.metric("Diff. vs Moyenne", f"{diff_mean:.2f}", delta=f"{diff_mean:.2f}")
                with col3:
                    st.metric("Min Dataset", f"{actual_min:.2f}")
                with col4:
                    st.metric("Max Dataset", f"{actual_max:.2f}")
                
                # Graphique
                fig = go.Figure()
                
                fig.add_trace(go.Box(
                    y=df[target_col],
                    name='Distribution Dataset',
                    marker_color='lightblue'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[0],
                    y=[prediction[0]],
                    mode='markers',
                    name='Votre Prédiction',
                    marker=dict(size=20, color='red', symbol='star')
                ))
                
                fig.update_layout(
                    title=f"Position de votre prédiction",
                    yaxis_title=target_col,
                    height=400
                )
                
                st.plotly_chart(fig, width="stretch")
                
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

# ----------------------------
# MODE 2: COMPARAISON
# ----------------------------
elif app_mode == "📊 Comparaison de Modèles":
    
    st.subheader("📊 Comparaison des Modèles")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Configuration")
        
        st.info(f"""
        **Targets disponibles:**
        - {', '.join([f'**{t}**' for t in available_targets])}
        """)
        
        target_col = st.selectbox("Variable cible", available_targets)
        
        # Charger les modèles pour ce target
        models = load_models_for_target(target_col)
        
        if not models:
            st.error(f"❌ Aucun modèle disponible pour {target_col}!")
            st.info("💡 Exécutez `python train_multitarget_models.py`")
            st.stop()
        
        selected_models = st.multiselect(
            "Modèles à comparer",
            list(models.keys()),
            default=list(models.keys())
        )
        
        n_samples = st.slider(
            "Échantillons à tester",
            min_value=10,
            max_value=min(1000, len(df)),
            value=100,
            step=10
        )
    
    with col2:
        st.markdown("### 📋 Modèles Disponibles")
        
        models_info = []
        for model_name in models.keys():
            models_info.append({
                "Modèle": model_name,
                "Type": type(models[model_name]).__name__,
                "Target": target_col,
                "Statut": "✅ Prêt"
            })
        
        st.dataframe(pd.DataFrame(models_info), width="stretch")
    
    st.markdown("---")
    
    if st.button("🚀 COMPARER LES MODÈLES"):
        
        if not selected_models:
            st.warning("⚠️ Sélectionnez au moins un modèle")
        else:
            sample_indices = np.random.choice(len(df), size=min(n_samples, len(df)), replace=False)
            df_sample = df.iloc[sample_indices]
            
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, model_name in enumerate(selected_models):
                status_text.text(f"⏳ Test de {model_name}...")
                
                try:
                    model = models[model_name]
                    
                    X_test, y_test, missing = prepare_features_for_model(df_sample, target_col, model)
                    
                    if missing:
                        st.warning(f"⚠️ {model_name}: {len(missing)} features manquantes - ignoré")
                        continue
                    
                    # CRITICAL FIX: Fill NaN values before prediction (match training preprocessing)
                    if X_test.isnull().sum().sum() > 0:
                        X_test = X_test.fillna(X_test.median())
                    
                    if y_test.isnull().sum() > 0:
                        valid_idx = y_test.notna()
                        X_test = X_test[valid_idx]
                        y_test = y_test[valid_idx]
                    
                    y_pred = model.predict(X_test)
                    
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results.append({
                        "Modèle": model_name,
                        "R² Score": r2,
                        "RMSE": rmse,
                        "MAE": mae,
                        "MSE": mse
                    })
                    
                except Exception as e:
                    st.error(f"❌ {model_name}: {str(e)[:100]}")
                
                progress_bar.progress((idx + 1) / len(selected_models))
            
            status_text.text("✅ Comparaison terminée!")
            
            if results:
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('R² Score', ascending=False)
                
                st.markdown("---")
                st.subheader("📊 Résultats")
                
                best_model = results_df.iloc[0]['Modèle']
                best_r2 = results_df.iloc[0]['R² Score']
                
                st.markdown(f"""
                <div class="prediction-box">
                    🏆 Meilleur: {best_model}<br>
                    R² = {best_r2:.4f}
                </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(
                    results_df.style.highlight_max(axis=0, subset=['R² Score'], color='lightgreen')
                                   .highlight_min(axis=0, subset=['RMSE', 'MAE', 'MSE'], color='lightgreen'),
                    width="stretch"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = go.Figure(data=[
                        go.Bar(
                            x=results_df['Modèle'],
                            y=results_df['R² Score'],
                            text=results_df['R² Score'].round(4),
                            textposition='auto',
                            marker_color='lightblue'
                        )
                    ])
                    fig1.update_layout(title="R² Score", height=400)
                    st.plotly_chart(fig1, width="stretch")
                
                with col2:
                    fig2 = go.Figure(data=[
                        go.Bar(
                            x=results_df['Modèle'],
                            y=results_df['RMSE'],
                            text=results_df['RMSE'].round(2),
                            textposition='auto',
                            marker_color='lightcoral'
                        )
                    ])
                    fig2.update_layout(title="RMSE", height=400)
                    st.plotly_chart(fig2, width="stretch")
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Télécharger CSV",
                    csv,
                    "comparaison_modeles.csv",
                    "text/csv"
                )
            else:
                st.error("❌ Aucun modèle n'a pu être testé")

# ----------------------------
# MODE 3: ANALYSE
# ----------------------------
elif app_mode == "📈 Analyse du Dataset":
    
    st.subheader("📈 Analyse Exploratoire")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📏 Lignes", f"{df.shape[0]:,}")
    with col2:
        st.metric("📊 Colonnes", df.shape[1])
    with col3:
        st.metric("🔢 Numériques", len(numeric_columns))
    with col4:
        st.metric("❓ Manquantes", df.isnull().sum().sum())
    
    st.markdown("---")
    
    st.subheader("🔍 Aperçu")
    
    n_rows = st.slider("Nombre de lignes", 5, 100, 10)
    st.dataframe(df.head(n_rows), width="stretch")
    
    st.markdown("---")
    
    st.subheader("📊 Statistiques")
    st.dataframe(df[numeric_columns].describe(), width="stretch")
    
    st.markdown("---")
    
    st.subheader("📈 Distribution")
    
    selected_var = st.selectbox("Variable à analyser", numeric_columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(df, x=selected_var, nbins=50)
        st.plotly_chart(fig_hist, width="stretch")
    
    with col2:
        fig_box = px.box(df, y=selected_var)
        st.plotly_chart(fig_box, width="stretch")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    🏪 Global Superstore ML Deployment<br>
    <small>Projet ML avec Streamlit</small>
</div>
""", unsafe_allow_html=True)
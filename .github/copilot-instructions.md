# AI Coding Agent Instructions for Global Superstore ML Project

## Architecture Overview
This is a Streamlit web application for deploying and comparing regression models trained on the Global Superstore dataset. The app predicts "Profit" using features like Quantity, Discount, Shipping Cost, and encoded categorical variables.

**Key Components:**
- `app.py`: Main Streamlit interface with three modes (prediction, comparison, analysis)
- `src/model_loader.py`: Loads trained models from `models/` directory
- `src/preprocessor.py`: Preprocesses data to match training format (removes high-cardinality text columns, one-hot encodes categoricals <50 unique values, creates Avg_Unit_Price)
- `models/`: Directory containing trained `.pkl` model files
- `data/Global_Superstore_FIXED.csv`: Preprocessed dataset

## Critical Workflows
- **Launch App**: `streamlit run app.py` (opens at http://localhost:8501)
- **Train New Models**: Use `notebooks/store.ipynb` or `train_models.py`, save as `.pkl` in `models/`, update `model_loader.py` dict
- **Add Model to UI**: Edit `model_files` dict in `src/model_loader.py` with model name and filename
- **Debug Models**: Models must have `.predict()` method; use `src/model_loader.py` standalone to test loading

## Project Conventions
- **Model Storage**: All models saved as `.pkl` files in `models/` using `joblib.dump(model, 'models/model_name.pkl')`
- **Data Preprocessing**: Always use `src/preprocessor.py` functions to ensure feature consistency between training and inference
- **Caching**: Use `@st.cache_data` for data loading, `@st.cache_resource` for model loading
- **Feature Engineering**: Categorical columns with <50 unique values get one-hot encoded; high-cardinality text columns dropped
- **Target Variable**: Default target is "Profit"; models expect specific feature set defined in training scripts

## Integration Points
- **External Dependencies**: Relies on scikit-learn, XGBoost, LightGBM for models; Plotly for visualizations
- **Data Flow**: Raw CSV → `preprocessor.py` → Encoded features → Model prediction → Streamlit display
- **Model Compatibility**: Ensure new models are trained on same preprocessed features; use `prepare_features_for_model()` for inference

## Common Patterns
- **Model Loading**: Check `hasattr(model, 'predict')` before using
- **Error Handling**: Wrap data loading in try/except with user-friendly Streamlit messages
- **UI Layout**: Use `st.columns()` for side-by-side displays; `st.expander()` for optional sections
- **Visualization**: Prefer Plotly over Matplotlib for interactive charts in Streamlit

## Key Files to Reference
- [app.py](app.py#L100-L150): Main prediction logic and UI modes
- [src/model_loader.py](src/model_loader.py#L20-L40): Model registry and loading
- [src/preprocessor.py](src/preprocessor.py#L70-L90): One-hot encoding and feature creation
- [train_models.py](train_models.py#L25-L35): Training feature selection and preprocessing</content>
<parameter name="filePath">c:\Users\sbiss\OneDrive - ESPRIT\Desktop\ML_Streamlit - Copy\projet_ml_comparaison\.github\copilot-instructions.md
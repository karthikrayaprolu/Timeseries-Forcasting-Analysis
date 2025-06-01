import pandas as pd
import numpy as np
import joblib
import tempfile
import os
import time
import traceback
from fastapi import Request, Depends
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse


from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
from prophet import Prophet
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.layers import Input, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from io import BytesIO
import uuid
from pymongo import MongoClient
from scipy.stats import spearmanr

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import lightgbm as lgb

# Define your MongoDB URI here, e.g., "mongodb://localhost:27017/"


# from auth import router as auth_router

app = FastAPI()
# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'csv'}

# Global variables
current_dataset = None
processed_data = None
trained_models = {}



def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_numeric_string(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    value = str(value).strip()
    value = value.replace('?', '').replace(',', '').replace('$', '').replace('%', '')
    try:
        return float(value)
    except ValueError:
        return np.nan

def clean_numeric_column(df: pd.DataFrame, col: str) -> pd.Series:
    """Enhanced numeric column cleaning with better handling of different data types"""
    try:
        series = df[col].copy()
        
        # If already numeric, just handle NaN and infinite values
        if pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors='coerce')
            series = series.replace([np.inf, -np.inf], np.nan)
            return series
        
        # Handle object/string columns
        if series.dtype == 'object':
            # Convert to string first
            series = series.astype(str)
            
            # Handle common patterns
            series = series.str.strip()  # Remove whitespace
            series = series.replace(['', 'NULL', 'null', 'NaN', 'nan', 'N/A', 'n/a'], np.nan)
            
            # Remove common non-numeric characters but preserve decimals and negative signs
            series = series.str.replace(r'[$,%€£¥]', '', regex=True)  # Currency symbols
            series = series.str.replace(r'[^\d.-]', '', regex=True)   # Keep only digits, dots, and hyphens
            
            # Handle multiple dots (keep only first one)
            series = series.apply(lambda x: '.'.join(x.split('.')[:2]) if isinstance(x, str) and '.' in x else x)
            
            # Convert to numeric
            series = pd.to_numeric(series, errors='coerce')
            
        else:
            # For other data types, try direct conversion
            series = pd.to_numeric(series, errors='coerce')
        
        # Replace infinite values with NaN
        series = series.replace([np.inf, -np.inf], np.nan)
        
        return series
        
    except Exception as e:
        print(f"Error cleaning column {col}: {str(e)}")
        # Return original series if cleaning fails
        return df[col]

def validate_target_variable(df: pd.DataFrame, target_col: str) -> tuple[pd.Series, str]:
    """Validate and clean target variable, return cleaned series and status message"""
    if target_col not in df.columns:
        raise ValueError(f"Target variable '{target_col}' not found in dataset")
    
    original_series = df[target_col].copy()
    cleaned_series = clean_numeric_column(df, target_col)
    
    # Remove NaN values
    cleaned_series = cleaned_series.dropna()
    
    if len(cleaned_series) == 0:
        raise ValueError(f"Target variable '{target_col}' has no valid numeric data")
    
    # Check variance
    unique_values = len(cleaned_series.unique())
    variance = cleaned_series.var()
    
    if unique_values <= 1 or variance == 0:
        raise ValueError(f"Target variable '{target_col}' has no variance (all values are identical)")
    
    status_msg = f"Target variable cleaned: {len(original_series)} -> {len(cleaned_series)} valid values, variance: {variance:.4f}"
    
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class UserSignup(BaseModel):
    name: str
    email: str
    password: str

class DataConfig(BaseModel):
    timeColumn: str
    targetVariable: str
    frequency: str
    features: List[str] = Field(default_factory=list)

    forecast_horizon: int =10
    def __init__(self, **data):
        super().__init__(**data)
        if 'forecast_horizon' not in data:
            print("[WARNING] forecast_horizon was not provided by the user.")
        else:
            print(f"[Success] User specified forecast_horizon: {data['forecast_horizon']}")

    modelType: str = "ensemble"
    hyperparameterTuning: bool = False
    ensembleLearning: bool = False
    transferLearning: bool = False

    ensembleModels: List[str] = Field(default_factory=lambda: ["arima", "lstm", "xgboost"])
    ensembleMethod: str = "voting"
    ensembleWeights: Optional[List[float]] = None
    sourceModelId: Optional[str] = None
    modelLevel: str = "balanced" 
    order: Tuple[int, int, int] = (1, 1, 1)
    seasonal_order: Optional[Tuple[int, int, int, int]] = None
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    seasonality_mode: str = "additive"

    units: int = 50
    dropout: float = 0.2
    epochs: int = 100
    batch_size: int = 32
    sequence_length: int = 10

    n_estimators: int = 100
    max_depth: int = 10
    learning_rate: float = 0.1
    gamma: float = 0
    min_child_weight: float = 0
    subsample: float = 0.8

    showCorrelationMap: bool = True
    correlationThreshold: float = 0.3  # Default threshold
    maxFeaturesToShow: int = 10  # Limit number of features in correlation map
    tuningMode: str = "balanced"  # new field: basic, fast, high_accuracy
    advancedParams: Optional[Dict[str, Any]] = None
    
    # ETS parameters
    trend: str = "add"
    seasonal: str = "add"
    seasonal_periods: int = 12
    
    # LightGBM parameters
    num_leaves: int = 31
    learning_rate: float = 0.1
    n_estimators: int = 100

class CorrelationRequest(BaseModel):
    targetVariable: str

class ExportConfig(BaseModel):
    data: Dict[str, Any]
    format: str
class EnsembleTimeSeriesModel(BaseEstimator, RegressorMixin):
    def __init__(self, models, weights=None, method='voting'):
        self.models = models
        self.weights = weights if weights is not None else [1/len(models)] * len(models)
        self.method = method
        self.scalers = {}
        self.feature_names_ = None
        
    def fit(self, X, y):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        y = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()
        
        for i, model in enumerate(self.models):
            try:
                if isinstance(model, SARIMAX) or isinstance(model, SARIMAXResultsWrapper):
                    continue
                elif isinstance(model, Prophet):
                    continue
                elif isinstance(model, Model):
                    sequence_length = getattr(model, 'sequence_length', 10)
                    self.scalers[i] = MinMaxScaler()
                    if processed_data and 'target' in processed_data['config']:
                        target_col = processed_data['config']['target']
                        if target_col not in X.columns:
                            X[target_col] = y

                    X_scaled = self.scalers[i].fit_transform(X)
                    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

                    sequences, targets = prepare_sequence_data(
                        X_scaled_df,
                        processed_data['config']['target'],
                        sequence_length
                    )
                    self.feature_names_ = list(X.columns)
                    model.fit(
                        sequences, 
                        targets,
                        epochs=100,
                        batch_size=32,
                        verbose=0,
                        callbacks=[
                            EarlyStopping(monitor='loss', patience=10),
                            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)
                        ]
                    )
                else:
                    self.scalers[i] = MinMaxScaler()
                    X_scaled = self.scalers[i].fit_transform(X)
                    model.fit(X_scaled, y)

            except Exception as e:
                print(f"Error fitting model {i} ({type(model).__name__}): {str(e)}")
                traceback.print_exc()
        
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        all_predictions = []
        successful_models = []

        if hasattr(self, 'feature_names_'):
            # Add missing features with default 0
            for feature in self.feature_names_:
                if feature not in X.columns:
                    X[feature] = 0

            # Drop extra features that were not seen during training
            X = X[self.feature_names_]


        for i, model in enumerate(self.models):
            try:
                pred = None  # Safe default to avoid uninitialized use

                if isinstance(model, SARIMAXResultsWrapper):
                    steps = len(X)
                    n_periods = processed_data['config'].get('forecast_horizon', 30)
                    pred = model.get_forecast(steps=n_periods).predicted_mean


                elif isinstance(model, Prophet):
                    last_date = processed_data['data'].index[-1]
                    n_periods = processed_data['config'].get('forecast_horizon', 30)
                    freq_map = {
                        'daily': 'D',
                        'weekly': 'W-MON',
                        'monthly': 'MS'
                    }
                    freq = freq_map.get(processed_data['config']['frequency'], 'D')

                    # Generate future dates
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(1, unit='d'), periods=n_periods, freq=freq)
                    future_df = pd.DataFrame({'ds': future_dates})

                    # Predict with Prophet
                    forecast = model.predict(future_df)
                    pred = forecast['yhat'].values

                elif isinstance(model, Model):  # LSTM
                    sequence_length = getattr(model, 'sequence_length', 10)
                    scaler = self.scalers.get(i)
                    if scaler is not None:
                        X_scaled = scaler.transform(X)
                        X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
                        forecast = []
                        current_sequence = X_scaled_df.iloc[-sequence_length:].values.reshape(1, sequence_length, -1)

                        for _ in range(processed_data['config'].get('forecast_horizon', 30)):
                            next_pred = model.predict(current_sequence, verbose=0)[0][0]
                            forecast.append(next_pred)

                            # Build new input sequence
                            next_input = current_sequence[:, 1:, :].copy()  # Drop first time step
                            next_input = np.append(next_input, [[[next_pred] * X_scaled_df.shape[1]]], axis=1)  # Repeat pred across features
                            current_sequence = next_input

                        if current_sequence is None or current_sequence.shape[1] == 0:
                            raise ValueError("Not enough data for LSTM prediction")
                        pred = np.array(forecast)
                        pred = pred[:len(X)]
                        dummy = np.zeros((len(pred), X.shape[1]))
                        target_idx = X.columns.get_loc(processed_data['config']['target'])
                        dummy[:, target_idx] = pred
                        pred = scaler.inverse_transform(dummy)[:, target_idx]

                else:  
                    scaler = self.scalers.get(i)
                    if scaler is not None:
                        X_scaled = scaler.transform(X)
                        pred = model.predict(X_scaled)
                    else:
                        pred = model.predict(X.to_numpy())

                if pred is None:
                    raise ValueError("Prediction returned None")

                pred = np.asarray(pred).flatten()
                if len(pred) > 0 and not np.any(np.isnan(pred)):
                    all_predictions.append(pred)
                    successful_models.append(i)
                    print(f"Model {i} ({type(model).__name__}) - prediction success. Length: {len(pred)}")
                else:
                    raise ValueError("Prediction invalid or contains NaN values")

            except Exception as e:
                print(f"Error in model {i} ({type(model).__name__}) prediction: {str(e)}")
                traceback.print_exc()
                continue

        if not all_predictions:
            raise ValueError("No valid predictions from any model")

    # Truncate predictions to common length
        min_length = min(len(p) for p in all_predictions)
        predictions = np.array([p[:min_length] for p in all_predictions])
        return np.mean(predictions, axis=0)
    

def predict_future(self, steps: int):
    if not hasattr(self, 'feature_names_'):
        raise ValueError("Model is not fitted with features")

    last_input = processed_data['data'][self.feature_names_].iloc[-1]
    future_inputs = pd.DataFrame([last_input.copy()] * steps)

    for col in future_inputs.columns:
        if future_inputs[col].dtype == 'object':
            future_inputs[col] = 0  # default for unseen categories

    return self.predict(future_inputs)

def get_default_params(model_type: str, tuning_mode: str) -> Dict[str, Any]:
    presets = {
        "lstm": {
            "basic": {"units": 32, "dropout": 0.1, "batch_size": 32, "epochs": 50, "sequence_length": 10},
            "balanced": {"units": 64, "dropout": 0.2, "batch_size": 32, "epochs": 100, "sequence_length": 10},
            "high_accuracy": {"units": 128, "dropout": 0.3, "batch_size": 16, "epochs": 150, "sequence_length": 20}
        },
        "random_forest": {
            "basic": {"n_estimators": 50, "max_depth": 6},
            "balanced": {"n_estimators": 100, "max_depth": 10},
            "high_accuracy": {"n_estimators": 200, "max_depth": 15}
        },
        "xgboost": {
            "basic": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.3},
            "balanced": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
            "high_accuracy": {"n_estimators": 200, "max_depth": 9, "learning_rate": 0.05}
        },
        "arima": {
            "basic": {"order": (1,1,1)},
            "balanced": {"order": (2,1,2)},
            "high_accuracy": {"order": (3,1,3)}
        },
        "prophet": {
            "basic": {"changepoint_prior_scale": 0.01, "seasonality_prior_scale": 5.0},
            "balanced": {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0},
            "high_accuracy": {"changepoint_prior_scale": 0.1, "seasonality_prior_scale": 15.0}
        },
        "ets": {
            "basic": {"trend": "add", "seasonal": "add", "seasonal_periods": 7},
            "balanced": {"trend": "add", "seasonal": "add", "seasonal_periods": 12},
            "high_accuracy": {"trend": "mul", "seasonal": "mul", "seasonal_periods": 12}
        },
        "lightgbm": {
            "basic": {"num_leaves": 15, "learning_rate": 0.3, "n_estimators": 50},
            "balanced": {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100},
            "high_accuracy": {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 200}
        }
    }
    return presets.get(model_type, {}).get(tuning_mode, {})

def load_pretrained_models():
    """Improved model loading with better metadata handling"""
    if not os.path.exists("models"):
        return
        
    for filename in os.listdir("models"):
        try:
            model_id = os.path.splitext(filename)[0]
            filepath = os.path.join("models", filename)
            
            if filename.endswith(".pkl"):
                model_data = joblib.load(filepath)
                if isinstance(model_data, dict):
                    trained_models[model_id] = model_data
                else:
                    # Handle case where only model object was saved
                    trained_models[model_id] = {
                        'model': model_data,
                        'params': {},
                        'config': {},
                        'target': 'unknown',
                        'metrics': {}
                    }
                    
            elif filename.endswith((".h5", ".keras")):
                model = tf.keras.models.load_model(filepath)
                # Try to load associated metadata
                metadata_path = os.path.join("models", f"{model_id}_meta.pkl")
                if os.path.exists(metadata_path):
                    metadata = joblib.load(metadata_path)
                else:
                    metadata = {
                        'params': {},
                        'config': {},
                        'target': 'unknown',
                        'metrics': {}
                    }
                
                trained_models[model_id] = {
                    'model': model,
                    **metadata
                }
                
        except Exception as e:
            print(f"Error loading model {filename}: {str(e)}")
            continue
load_pretrained_models() 


@app.get("/api/models")
async def get_available_models(request: Request):
    try:
        user_id = request.headers.get("X-User-Id", "anonymous")
        
        if not trained_models:
            load_pretrained_models()  # Attempt to reload if empty
            
        models = []
        for model_id, model_info in trained_models.items():
            try:
                # Skip models that don't belong to the current user
                if model_info.get('user_id') != user_id:
                    continue
                    
                # Validate model info structure
                if not isinstance(model_info, dict):
                    continue
                    
                if 'model' not in model_info:
                    continue
                    
                models.append({
                    'id': model_id,
                    'type': model_info.get('config', {}).get('modelType', 'unknown').lower(),
                    'target': model_info.get('target', 'unknown'),
                    'metrics': model_info.get('metrics', {})
                })
            except Exception as e:
                print(f"Error processing model {model_id}: {str(e)}")
                continue
                
        return models
        
    except Exception as e:
        print(f"Error in /api/models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to load available models"
        )
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    global current_dataset
    current_dataset = pd.read_csv(filepath)
    if current_dataset is None or current_dataset.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty")
    
    return {
        "message": "File uploaded successfully",
        "tables": [file.filename.replace('.csv', '')],
        "stats": {
            "rows": len(current_dataset),
            "columns": list(current_dataset.columns)
        }
    }

@app.get("/api/columns")
async def get_columns():
    if current_dataset is None:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    return {"columns": list(current_dataset.columns)}
@app.post("/api/correlations")
async def get_correlations(config: CorrelationRequest) -> Dict[str, float]:
    global current_dataset
    
    if current_dataset is None:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    
    df = current_dataset.copy()
    
    try:
        # Step 1: Verify target variable exists first
        if config.targetVariable not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target variable '{config.targetVariable}' not found in dataset"
            )
        
        print(f"Original target variable '{config.targetVariable}' has {len(df[config.targetVariable].unique())} unique values")
        print(f"Target sample values: {df[config.targetVariable].head().tolist()}")
        
        # Step 2: Clean target variable first and check variance before cleaning other columns
        target_series = df[config.targetVariable].copy()
        
        # Clean target variable
        if target_series.dtype == 'object':
            # Remove non-numeric characters and convert
            target_series = target_series.astype(str).str.replace(r'[^0-9.-]', '', regex=True)
            target_series = pd.to_numeric(target_series, errors='coerce')
        else:
            target_series = pd.to_numeric(target_series, errors='coerce')
        
        # Remove NaN values from target
        target_series = target_series.dropna()
        
        if len(target_series) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Target variable '{config.targetVariable}' has no valid numeric data after cleaning"
            )
        
        # Check variance in cleaned target
        if len(target_series.unique()) <= 1 or target_series.var() == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Target variable '{config.targetVariable}' has no variance after cleaning (all values are identical)"
            )
        
        print(f"Cleaned target variable has {len(target_series.unique())} unique values")
        print(f"Target variance: {target_series.var()}")
        
        # Step 3: Clean all other columns
        cleaned_df = pd.DataFrame(index=df.index)
        cleaned_df[config.targetVariable] = target_series
        
        for col in df.columns:
            if col == config.targetVariable:
                continue
                
            try:
                series = df[col].copy()
                
                # Try direct numeric conversion first
                if series.dtype in ['int64', 'float64']:
                    cleaned_series = pd.to_numeric(series, errors='coerce')
                else:
                    # Clean string data
                    if series.dtype == 'object':
                        # Remove common non-numeric characters
                        series = series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
                        series = series.replace('', np.nan)
                        cleaned_series = pd.to_numeric(series, errors='coerce')
                    else:
                        cleaned_series = pd.to_numeric(series, errors='coerce')
                
                # Only keep columns with some valid numeric data and variance
                cleaned_series = cleaned_series.dropna()
                if len(cleaned_series) > 0 and len(cleaned_series.unique()) > 1:
                    # Align with target variable index
                    common_index = target_series.index.intersection(cleaned_series.index)
                    if len(common_index) > 1:  # Need at least 2 points for correlation
                        cleaned_df.loc[common_index, col] = cleaned_series.loc[common_index]
                        
            except Exception as e:
                print(f"Warning: Could not clean column {col}: {str(e)}")
                continue
        
        # Step 4: Remove rows where target is NaN and ensure we have enough data
        cleaned_df = cleaned_df.dropna(subset=[config.targetVariable])
        
        if len(cleaned_df) < 2:
            raise HTTPException(
                status_code=400,
                detail="Not enough valid data points for correlation analysis (need at least 2)"
            )
        
        print(f"Final dataset shape: {cleaned_df.shape}")
        print(f"Columns with data: {cleaned_df.columns.tolist()}")
        
        # Step 5: Calculate correlations
        correlations = {}
        target_data = cleaned_df[config.targetVariable]
        
        for col in cleaned_df.columns:
            if col == config.targetVariable:
                continue
                
            try:
                feature_data = cleaned_df[col].dropna()
                
                # Find common indices
                common_idx = target_data.index.intersection(feature_data.index)
                if len(common_idx) < 2:
                    continue
                    
                target_common = target_data.loc[common_idx]
                feature_common = feature_data.loc[common_idx]
                
                # Remove any remaining NaN pairs
                valid_mask = ~(target_common.isna() | feature_common.isna())
                target_clean = target_common[valid_mask]
                feature_clean = feature_common[valid_mask]
                
                if len(target_clean) < 2 or len(feature_clean.unique()) <= 1:
                    continue
                
                # Calculate both Pearson and Spearman correlations
                try:
                    pearson_corr = target_clean.corr(feature_clean, method='pearson')
                    spearman_corr = target_clean.corr(feature_clean, method='spearman')
                    
                    # Use the correlation with larger absolute value
                    if pd.isna(pearson_corr) and pd.isna(spearman_corr):
                        correlations[col] = 0.0
                    elif pd.isna(pearson_corr):
                        correlations[col] = round(float(spearman_corr), 4)
                    elif pd.isna(spearman_corr):
                        correlations[col] = round(float(pearson_corr), 4)
                    else:
                        if abs(pearson_corr) >= abs(spearman_corr):
                            correlations[col] = round(float(pearson_corr), 4)
                        else:
                            correlations[col] = round(float(spearman_corr), 4)
                            
                except Exception as corr_error:
                    print(f"Correlation calculation failed for {col}: {str(corr_error)}")
                    correlations[col] = 0.0
                    
            except Exception as e:
                print(f"Error processing column {col}: {str(e)}")
                correlations[col] = 0.0
                continue
        
        if not correlations:
            raise HTTPException(
                status_code=400,
                detail="No valid correlations could be calculated. Please check your data quality."
            )
        
        # Sort by absolute correlation value
        sorted_correlations = dict(
            sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)
        )
        
        print(f"Calculated {len(sorted_correlations)} correlations")
        print(f"Top correlations: {dict(list(sorted_correlations.items())[:5])}")
        
        return sorted_correlations
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"Unexpected error in correlation calculation: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to compute correlations: {str(e)}"
        )
@app.post("/api/process")
async def process_data(config: DataConfig):
    if current_dataset is None:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    
    df = current_dataset.copy()
    try:
        df[config.timeColumn] = pd.to_datetime(df[config.timeColumn])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting time column: {str(e)}")
    
    if config.targetVariable not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target variable '{config.targetVariable}' not found")
    
    df[config.targetVariable] = clean_numeric_column(df, config.targetVariable)
    
    for feature in config.features:
        if feature in df.columns:
            df[feature] = clean_numeric_column(df, feature)
    
    # Causal Feature Engineering: Lagged values and shifted rolling statistics
    for lag in [1, 2, 3]:
        df[f'{config.targetVariable}_lag{lag}'] = df[config.targetVariable].shift(lag)
    df[f'{config.targetVariable}_rolling_mean'] = df[config.targetVariable].rolling(window=3).mean().shift(1)
    df[f'{config.targetVariable}_rolling_std'] = df[config.targetVariable].rolling(window=3).std().shift(1)
    
    agg_dict = {}

    if config.targetVariable in df.columns:
        agg_dict[config.targetVariable] = 'mean'

    for feature in config.features:
        if feature in df.columns:
            agg_dict[feature] = 'mean'

    for lag in [1, 2, 3]:
        lag_col = f"{config.targetVariable}_lag{lag}"
        if lag_col in df.columns:
            agg_dict[lag_col] = 'mean'

    if f'{config.targetVariable}_rolling_mean' in df.columns:
        agg_dict[f'{config.targetVariable}_rolling_mean'] = 'mean'
    if f'{config.targetVariable}_rolling_std' in df.columns:
        agg_dict[f'{config.targetVariable}_rolling_std'] = 'mean'

    # ✅ Then apply it
    df = df.groupby(config.timeColumn).agg(agg_dict).reset_index()

    
    df.set_index(config.timeColumn, inplace=True)
    df.sort_index(inplace=True)
    
    df = df.ffill().bfill()
    
    if config.frequency == 'daily':
        df = df.resample('D').mean().ffill().bfill()
    elif config.frequency == 'weekly':
        df = df.resample('W-MON').mean().ffill().bfill()
    elif config.frequency == 'monthly':
        df = df.resample('MS').mean().ffill().bfill()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported frequency: {config.frequency}")
    
    global processed_data
    processed_data = {
        'data': df,
        'config': {
            'time_column': config.timeColumn,
            'target': config.targetVariable,
            'frequency': config.frequency,
            'forecast_horizon': config.forecast_horizon
        },
        'features': config.features + [f'{config.targetVariable}_lag{lag}' for lag in [1, 2, 3]] + 
                    [f'{config.targetVariable}_rolling_mean', f'{config.targetVariable}_rolling_std']
    }
    correlation_matrix = df.corr().round(2).fillna(0).to_dict()
    return {
        "message": "Data processed successfully",
        "preview": {
            "rows": len(df),
            "start_date": df.index.min().strftime('%Y-%m-%d'),
            "end_date": df.index.max().strftime('%Y-%m-%d'),
            "columns": df.columns.tolist(),
            "sample": df.head(5).to_dict(orient='records')
        },
        "correlationMatrix": correlation_matrix
    }

def prepare_sequence_data(data, target_col, sequence_length):
    sequences = []
    targets = []
    number_of_features = data.shape[1]
    for i in range(len(data) - sequence_length):
        sequence = data.iloc[i:i + sequence_length].values
        if target_col in data.columns:
            target = data[target_col].iloc[i + sequence_length]
        else:
            raise KeyError(f"Target column '{target_col}' not found in data with columns: {data.columns.tolist()}")

        sequences.append(sequence)
        targets.append(target)
    return np.array(sequences), np.array(targets)

   
def evaluate_model(model, train, test, model_type):
    try:
        actuals = test[processed_data['config']['target']].values
        
        if model_type == 'arima':
            predictions = model.get_forecast(len(test)).predicted_mean
        elif model_type == 'prophet':
            future_dates = pd.DataFrame({'ds': test.index})
            forecast = model.predict(future_dates)
            predictions = forecast['yhat'].values
        elif model_type == 'lstm':
            sequence_length = model.input_shape[1] if hasattr(model, 'input_shape') else 10
            ensemble = getattr(model, '_ensemble', None)
            scaler = None
            if ensemble:
                model_idx = next((i for i, m in enumerate(ensemble.models) if m is model), None)
                if model_idx is not None:
                    scaler = ensemble.scalers.get(model_idx)
            
            if scaler is None:
                scaler = MinMaxScaler()
                train_data = pd.concat([train, test])
                scaler.fit(train_data)
            
            test_df = pd.DataFrame(test, index=test.index, columns=test.columns)
            scaled_test = scaler.transform(test_df)
            scaled_test_df = pd.DataFrame(scaled_test, index=test_df.index, columns=test_df.columns)
            
            X_test, _ = prepare_sequence_data(
                scaled_test_df,
                processed_data['config']['target'],
                sequence_length
            )
            
            if len(X_test) > 0:
                predictions = model.predict(X_test, verbose=0)
                predictions = predictions.flatten()
                dummy = np.zeros((len(predictions), test.shape[1]))
                target_idx = test.columns.get_loc(processed_data['config']['target'])
                dummy[:, target_idx] = predictions
                predictions = scaler.inverse_transform(dummy)[:, target_idx]
                predictions = predictions[:len(test)]
            else:
                raise ValueError("Not enough data points for sequence prediction")
        
        elif model_type == 'ets':
            predictions = model.forecast(len(test))
            
        elif model_type == 'lightgbm':
            X_test = test.drop(columns=[processed_data['config']['target']])
            predictions = model.predict(X_test.values)
                
        else:
            ensemble = getattr(model, '_ensemble', None)
            scaler = None
            if ensemble:
                model_idx = next((i for i, m in enumerate(ensemble.models) if m is model), None)
                if model_idx is not None:
                    scaler = ensemble.scalers.get(model_idx)
            
            if scaler is not None:
                X_test_scaled = scaler.transform(test)
                predictions = model.predict(X_test_scaled)
            else:
                X_test = test.drop(columns=[processed_data['config']['target']])
                predictions = model.predict(X_test.to_numpy())
        
        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()
        
        if len(predictions) == 0 or len(actuals) == 0:
            raise ValueError("Empty predictions or actuals")
            
        min_length = min(len(predictions), len(actuals))
        predictions = predictions[:min_length]
        actuals = actuals[:min_length]
        
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        predictions = predictions[mask]
        actuals = actuals[mask]
        
        if len(predictions) == 0 or len(actuals) == 0:
            raise ValueError("No valid predictions after filtering NaN values")
        
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        mape = np.mean(np.abs((actuals - predictions) / np.where(np.abs(actuals) < 1e-10, 1, actuals))) * 100
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape)
        }
    except Exception as e:
        print(f"Error in evaluate_model for {model_type}: {str(e)}")
        traceback.print_exc()
        return {
            'mse': float('inf'),
            'rmse': float('inf'),
            'mae': float('inf'),
            'mape': float('inf'),
            'error': str(e)
        }

def get_seasonal_order(frequency):
    if frequency == 'daily':
        return (1,1,1,7)
    elif frequency == 'weekly':
        return (1,1,1,4)
    elif frequency == 'monthly':
        return (1,1,1,12)
    else:
        return (0,0,0,0)

def train_arima(train, test, config):
    frequency = processed_data['config']['frequency']
    seasonal_order = get_seasonal_order(frequency)
    params = {
        'order': config.get('order', (1,1,1)),
        'seasonal_order': config.get('seasonal_order', seasonal_order)
    }
    train_data = train[processed_data['config']['target']]
    model = SARIMAX(train_data, order=params['order'], seasonal_order=params['seasonal_order'])
    fitted_model = model.fit(disp=False)
    return params, fitted_model

def train_prophet(train, test, config):
    params = {
        'changepoint_prior_scale': config.get('changepoint_prior_scale', 0.05),
        'seasonality_prior_scale': config.get('seasonality_prior_scale', 10),
        'seasonality_mode': config.get('seasonality_mode', 'additive')
    }
    model = Prophet(**params)
    df_prophet = pd.DataFrame({
        'ds': train.index,
        'y': train[processed_data['config']['target']]
    })
    model.fit(df_prophet)
    return params, model

def train_lstm(train, config):
    # Use config parameters directly (they will be either from presets or custom)
    params = {
        'units': config.get('units', 50),
        'dropout': config.get('dropout', 0.2),
        'epochs': config.get('epochs', 100),
        'batch_size': config.get('batch_size', 32),
        'sequence_length': config.get('sequence_length', 10)
    }
    print(params)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(train)
    scaled_df = pd.DataFrame(scaled_data, index=train.index, columns=train.columns)
    
    X, y = prepare_sequence_data(
        scaled_df,
        processed_data['config']['target'],
        params['sequence_length']
    )
    
    input_layer = Input(shape=(X.shape[1], X.shape[2]))
    x = Bidirectional(LSTM(params['units'], return_sequences=True))(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(params['dropout'])(x)
    x = LSTM(params['units'] // 2)(x)
    x = BatchNormalization()(x)
    x = Dropout(params['dropout'])(x)
    output_layer = Dense(1)(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(), loss='mse')
    
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    model.fit(
        X, y,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )
    
    model.scaler = scaler
    model.sequence_length = params['sequence_length']
    
    return params, model

def train_random_forest(train, test, config):
    params = {
        'n_estimators': config.get('n_estimators', 100),
        'max_depth': config.get('max_depth', 10),
        'random_state': 42
    }
    print(params)
    model = RandomForestRegressor(warm_start=True,**params, n_jobs=-1)
    X_train = train.drop(columns=[processed_data['config']['target']])
    y_train = train[processed_data['config']['target']]
    model.fit(X_train.values, y_train)
    return params, model

def train_xgboost(train, test, config):
    params = {
        'n_estimators': config.get('n_estimators', 100),
        'max_depth': config.get('max_depth', 6),
        'learning_rate': config.get('learning_rate', 0.1),
        'objective': 'reg:squarederror',
        'random_state': 42
    }
    print(params)
    model = XGBRegressor(**params, n_jobs=-1)
    X_train = train.drop(columns=[processed_data['config']['target']])
    y_train = train[processed_data['config']['target']]
    model.fit(X_train.values, y_train)
    return params, model

def train_ets(train, test, config):
    params = {
        'trend': config.get('trend', 'add'),
        'seasonal': config.get('seasonal', 'add'),
        'seasonal_periods': config.get('seasonal_periods', 12)
    }
    print(f"[ETS] Training with params: {params}")
    target = processed_data['config']['target']
    
    # Validate data
    train_data = train[target].dropna()
    if len(train_data) < 2 * params['seasonal_periods']:
        raise ValueError(f"Insufficient data points ({len(train_data)}) for ETS with seasonal_periods={params['seasonal_periods']}. Need at least {2 * params['seasonal_periods']}.")
    if train_data.isna().all():
        raise ValueError("Target column contains only NaN values")
    if train_data.var() < 1e-6:
        raise ValueError("Target column has near-zero variance, unsuitable for ETS")
    
    try:
        model = ETSModel(
            train_data,
            error='add',
            trend=params['trend'] if params['trend'] != 'none' else None,
            seasonal=params['seasonal'] if params['seasonal'] != 'none' else None,
            seasonal_periods=params['seasonal_periods']
        )
        fitted_model = model.fit(disp=False)
        print("[ETS] Model training completed successfully")
        return params, fitted_model
    except Exception as e:
        print(f"[ETS TRAINING ERROR] {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ETS training failed: {str(e)}")

def train_lightgbm(train, test, config):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': config.get('num_leaves', 31),
        'learning_rate': config.get('learning_rate', 0.1),
        'n_estimators': config.get('n_estimators', 100),
        'random_state': 42
    }
    print(params)
    target = processed_data['config']['target']
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # Validate data
    if X_train.empty:
        raise ValueError("No features available for LightGBM training")
    if y_train.isna().all():
        raise ValueError("Target column contains only NaN values")
    
    try:
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train.values, y_train)
        return params, model
    except Exception as e:
        print(f"[ERROR] LightGBM training failed: {str(e)}")
        raise

def get_hyperparameter_grid(model_type):
    if model_type == 'arima':
        return {
            'order': [(p,d,q) for p in range(5) for d in range(3) for q in range(5)],
            'seasonal_order': [get_seasonal_order(processed_data['config']['frequency'])]
        }
    elif model_type == 'prophet':
        return {
            'changepoint_prior_scale': [0.001, 0.01, 0.1],
            'seasonality_prior_scale': [0.01, 0.1, 1.0],
            'seasonality_mode': ['additive', 'multiplicative']
        }
    elif model_type == 'lstm':
        return {
            'units': [32, 50, 64],
            'dropout': [0.1, 0.2, 0.3],
            'batch_size': [16, 32, 64],
            'sequence_length': [5, 10, 20]
        }
    elif model_type == 'random_forest':
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [6, 10, 15]
        }
    elif model_type == 'xgboost':
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'gamma': [0, 2, 4],
            'min_child_weight': [0, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
    elif model_type == 'ets':
        return {
            'trend': ['add', 'mul', None],
            'seasonal': ['add', 'mul', None],
            'seasonal_periods': [7, 12, 24]
        }
    elif model_type == 'lightgbm':
        return {
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [50, 100, 200]
        }
        
    return {}

def tune_hyperparameters(model_type, train, test, config):
    param_grid = get_hyperparameter_grid(model_type)
    best_score = float('inf')
    best_params = None
    best_model = None
    
    for params in ParameterGrid(param_grid):
        try:
            config_copy = config.copy()
            config_copy.update(params)
            
            if model_type == 'arima':
                _, model = train_arima(train, test, config_copy)
            elif model_type == 'prophet':
                _, model = train_prophet(train, test, config_copy)
            elif model_type == 'lstm':
                _, model = train_lstm(train, config_copy)
            elif model_type == 'random_forest':
                _, model = train_random_forest(train, test, config_copy)
            elif model_type == 'xgboost':
                _, model = train_xgboost(train, test, config_copy)
                
            metrics = evaluate_model(model, train, test, model_type)
            score = metrics['rmse']
            
            if score < best_score:
                best_score = score
                best_params = params
                best_model = model
                
        except Exception as e:
            print(f"Error with params {params}: {str(e)}")
            continue
    
    return best_params, best_model

def apply_transfer_learning(source_model, train_data, config):
    print(f"[TL] source_model: {type(source_model)}")
    print(f"[TL] config: {config}")

    try:
        if isinstance(source_model, Model):  # better: hasattr(source_model, "predict") and "layers" in dir(...)
            print("[TL] Entering Keras TL branch")
            # Copy architecture using submodel
            truncated_model = Model(
                inputs=source_model.input,
                outputs=source_model.layers[-3].output  # or a layer name
            )
            truncated_model.trainable = False

            x = truncated_model.output
            x = Dense(32, activation='relu')(x)
            x = Dropout(0.2)(x)
            output = Dense(1)(x)

            new_model = Model(inputs=truncated_model.input, outputs=output)
            new_model.compile(optimizer=Adam(1e-4), loss='mse', metrics=['mae'])

            if hasattr(source_model, 'scaler'):
                new_model.scaler = source_model.scaler
            if hasattr(source_model, 'sequence_length'):
                new_model.sequence_length = source_model.sequence_length

            return new_model

        # For tree-based models (Random Forest, XGBoost)
        elif isinstance(source_model, (RandomForestRegressor, XGBRegressor)):
            if isinstance(source_model, RandomForestRegressor):
                new_model = RandomForestRegressor(
                    n_estimators=config.get('n_estimators', 50),
                    max_depth=source_model.max_depth,
                    warm_start=True,
                    random_state=42
                )
                if hasattr(source_model, 'estimators_'):
                    new_model.estimators_ = source_model.estimators_[:config.get('n_estimators', 50)]

            elif isinstance(source_model, XGBRegressor):
                new_model = XGBRegressor(
                    n_estimators=config.get('n_estimators', 50),
                    max_depth=source_model.max_depth,
                    learning_rate=0.01,
                    objective='reg:squarederror'
                )
                if hasattr(source_model, 'get_booster'):
                    new_model._Booster = source_model.get_booster()

            return new_model

    except Exception as e:
        print(f"Error in transfer learning: {str(e)}")
        traceback.print_exc()
        return None

def create_ensemble_model(train, test, config):
    print("Creating ensemble model...")
    base_models = []
    model_errors = []

    model_mapping = {
        'arima': 'ARIMA',
        'prophet': 'Prophet',
        'lstm': 'LSTM',
        'random_forest': 'RandomForest',
        'xgboost': 'XGBoost',
        'ets': 'ETS',
        'lightgbm': 'LightGBM'
    }
    # selected_models = [model_mapping.get(model, model) for model in config.get('ensembleModels', ['arima', 'lstm', 'xgboost'])]
    selected_models = [model_mapping.get(model, model) for model in config['ensembleModels']]
    print(selected_models)

    target = processed_data['config']['target']
    
    for model_type in selected_models:
        try:
            if model_type == 'ARIMA':
                _, model = train_arima(train, test, config)
        
            elif model_type == 'Prophet':
                _, model = train_prophet(train, test, config)
                
            elif model_type == 'LSTM':
                _, model = train_lstm(train, config)
                model.target_column = target
                
            elif model_type == 'RandomForest':
                X_train = train.drop(columns=[target])
                y_train = train[target]
                _, model = train_random_forest(train, test, config)
    
            elif model_type == 'XGBoost':
                X_train = train.drop(columns=[target])
                y_train = train[target]
                _, model = train_xgboost(train, test, config)
            
            elif model_type == 'ETS':
                _, model = train_ets(train, test, config)
                
            elif model_type == 'LightGBM':
                _, model = train_lightgbm(train, test, config)

            base_models.append((model_type.lower(), model))  # keep the lowercase name for traceability
            rmse = evaluate_model(model, train, test, model_type.lower()).get('rmse', np.inf)
            model_errors.append(rmse)

        except Exception as e:
            print(f"Error training {model_type}: {str(e)}")
            traceback.print_exc()
            continue
            
    if not base_models:
        raise ValueError("No models were successfully trained for the ensemble")
        
    weights = np.array([1 / (e + 1e-6) for e in model_errors])
    weights = weights / weights.sum()

    ensemble = EnsembleTimeSeriesModel(
    [model for _, model in base_models],
    weights=weights.tolist(),
    method='voting'
)
    ensemble.base_models = base_models
    return ensemble

@app.post("/api/train")
async def train_model(config: DataConfig, request: Request):
    user_id = request.headers.get("X-User-Id", "anonymous")
    print("[DEBUG] Incoming config:", config)

    if processed_data is None:
        raise HTTPException(status_code=404, detail="No processed data available. Please process data first.")
    
    user_forecast_horizon = config.forecast_horizon
    print(user_forecast_horizon)
   
    if user_forecast_horizon is None or user_forecast_horizon <= 0:
        forecast_horizon = 30  # Default
        print(f"[Warning] Invalid or missing forecast_horizon ({user_forecast_horizon}), using default: 30")
    else:
        forecast_horizon = int(user_forecast_horizon)
        print(f"[Success] User specified forecast_horizon: {forecast_horizon}")
    
    # Store it in processed_data for consistency
    processed_data['config']['forecast_horizon'] = int(user_forecast_horizon)
    print(user_forecast_horizon)
    
    df = processed_data['data']
    target = processed_data['config']['target']

    split_point = int(len(df) * 0.8)
    train = df.iloc[:split_point]
    test = df.iloc[split_point:]
    
    print(f"[CRITICAL] FINAL forecast_horizon being used: {forecast_horizon}")

    try:
        model = None
        params = {}
        original_forecast_horizon = forecast_horizon  # Preserve original value

        if config.modelLevel != 'custom':
            model_params = get_default_params(config.modelType, config.modelLevel)
            config_dict = config.model_dump()
            config_dict.update(model_params)
            config = DataConfig(**config_dict)
            config.forecast_horizon = forecast_horizon 
       
        config.forecast_horizon = original_forecast_horizon
        print(f"[Verification] Config forecast_horizon after recreation: {config.forecast_horizon}")
        
        if config.ensembleLearning:
            selected_models = config.ensembleModels
            print("[Ensemble] Selected models:", selected_models)
            if config.modelType not in selected_models:
                selected_models.append(config.modelType)

            ensemble = create_ensemble_model(
                train,
                test,
                {
                    'ensembleModels': selected_models,
                    'ensembleMethod': config.ensembleMethod,
                    'ensembleWeights': config.ensembleWeights,
                    'forecast_horizon': original_forecast_horizon,  # Pass it explicitly
                    **config.model_dump()
                }
            )

            X_train = train.drop(columns=[target])
            y_train = train[target]
            ensemble.fit(X_train, y_train)
            model = ensemble
            params = {
                'ensemble_models': selected_models,
                'method': config.ensembleMethod
            }

        transferred_model = None
        if config.transferLearning and config.sourceModelId:
            source_model_id = config.sourceModelId
            print("=== DEBUG TL BLOCK ===")
            print(f"transferLearning: {config.transferLearning}")
            print(f"sourceModelId: {config.sourceModelId}")
            print(f"trained_models keys: {list(trained_models.keys())}")

            if source_model_id not in trained_models:
                print(f"[TL] Source model {source_model_id} not found")
                raise HTTPException(status_code=404, detail="Source model not found")

            source_model_info = trained_models[source_model_id]
            source_type = source_model_info.get('config', {}).get('modelType', '').lower()
            target_type = config.modelType.lower()

            if source_type != target_type:
                print(f"[TL] Type mismatch: Source({source_type}) vs Target({target_type})")
                raise HTTPException(
                    status_code=400,
                    detail="Source model type must match target model type"
                )

            if source_type not in ['lstm', 'random_forest', 'xgboost']:
                print(f"[TL] Unsupported source model type: {source_type}")
                raise HTTPException(
                    status_code=400,
                    detail="Transfer learning not supported for this model type"
                )

            source_model = source_model_info['model']
            transferred_model = apply_transfer_learning(source_model, train, config.model_dump())

            if transferred_model is not None:
                print("[TL] Transfer learning succeeded")
                
                if isinstance(transferred_model, Model):  # Keras model
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(train)
                    scaled_df = pd.DataFrame(scaled_data, index=train.index, columns=train.columns)

                    sequence_length = getattr(transferred_model, 'sequence_length', 10)
                    X, y = prepare_sequence_data(scaled_df, target, sequence_length)

                    callbacks = [
                        EarlyStopping(patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(factor=0.5, patience=5)
                    ]

                    transferred_model.fit(
                        X, y,
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        validation_split=0.2,
                        callbacks=callbacks,
                        verbose=0
                    )

                    transferred_model.scaler = scaler
                    transferred_model.sequence_length = sequence_length

                else:  # Tree-based model
                    X_train = train.drop(columns=[target])
                    y_train = train[target]
                    transferred_model.fit(X_train, y_train)

                model = transferred_model
                params = {"source_model_id": source_model_id, "transferred": True}
            else:
                print("[TL] Transfer learning failed - falling back to regular training")

        # If no model from transfer learning or ensemble, do regular training
        if model is None:
            if config.modelType == 'arima':
                params, model = train_arima(train, test, config.model_dump())
            elif config.modelType == 'prophet':
                params, model = train_prophet(train, test, config.model_dump())
            elif config.modelType == 'lstm':
                params, model = train_lstm(train, config.model_dump())
            elif config.modelType == 'random_forest':
                params, model = train_random_forest(train, test, config.model_dump())
            elif config.modelType == 'xgboost':
                params, model = train_xgboost(train, test, config.model_dump())
            elif config.modelType == 'ets':
                params, model = train_ets(train, test, config.model_dump())
            elif config.modelType == 'lightgbm':
                params, model = train_lightgbm(train, test, config.model_dump())
            else:
                raise HTTPException(status_code=400, detail=f"Unknown model type: {config.modelType}")

        # Generate predictions on test set
        if config.ensembleLearning:
            X_test = test.copy()
            if hasattr(model, 'feature_names_'):
                for feature in model.feature_names_:
                    if feature not in X_test.columns:
                        X_test[feature] = 0
                X_test = X_test[model.feature_names_]
            else:
                X_test = test.drop(columns=[target])
            predictions = model.predict(X_test)
        else:
            if config.modelType == 'arima':
                predictions = model.get_forecast(len(test)).predicted_mean
            elif config.modelType == 'prophet':
                future_dates = pd.DataFrame({'ds': test.index})
                forecast = model.predict(future_dates)
                predictions = forecast['yhat'].values
            elif config.modelType == 'lstm':
                scaler = getattr(model, 'scaler', None)
                sequence_length = getattr(model, 'sequence_length', 10)
                if scaler is None:
                    scaler = MinMaxScaler()
                    concat_data = pd.concat([train, test])
                    scaler.fit(concat_data)
                    model.scaler = scaler

                scaled_test = scaler.transform(test)
                test_df = pd.DataFrame(scaled_test, index=test.index, columns=test.columns)
                X_test, _ = prepare_sequence_data(test_df, target, sequence_length)

                predictions = model.predict(X_test, verbose=0).flatten()
                dummy = np.zeros((len(predictions), test.shape[1]))
                target_idx = test.columns.get_loc(target)
                dummy[:, target_idx] = predictions
                predictions = scaler.inverse_transform(dummy)[:, target_idx]
                predictions = predictions[:len(test)]
            elif config.modelType == 'ets':
                try:
                    predictions = model.forecast(len(test))
                    if predictions is None or len(predictions) == 0:
                        raise ValueError("ETS forecast returned no predictions")
                    predictions = predictions.values.flatten()
                except Exception as e:
                    print(f"[ETS PREDICTION ERROR] {str(e)}")
                    traceback.print_exc()
                    raise HTTPException(status_code=500, detail=f"ETS test set prediction failed: {str(e)}")
            elif config.modelType == 'lightgbm':
                X_test = test.drop(columns=[target])
                predictions = model.predict(X_test.values)
            else:
                X_test = test.drop(columns=[target])
                predictions = model.predict(X_test.to_numpy())

        actuals = test[target].values
        metrics = evaluate_model(model, train, test, config.modelType if not config.ensembleLearning else 'ensemble')

        # Save model
        if not os.path.exists('models'):
            os.makedirs('models')

        model_id = f"{config.modelType}_{int(time.time())}"
        trained_models[model_id] = {
            'model': model,
            'params': params,
            'config': config.model_dump(),
            'train_data': train,
            'test_data': test,
            'target': target,
            'metrics': metrics,
            'user_id': user_id
        }

        if isinstance(model, Model):
            model.save(f"models/{model_id}.keras")
            joblib.dump({
                'params': params,
                'config': config.model_dump(),
                'target': target,
                'metrics': metrics,
                'user_id': user_id
            }, f"models/{model_id}_meta.pkl")
            print(f"Keras model saved at models/{model_id}.keras")
        else:
            joblib.dump({
                'model': model,
                'params': params,
                'config': config.model_dump(),
                'target': target,
                'metrics': metrics,
                'user_id': user_id
            }, f"models/{model_id}.pkl")
            print(f"Non-Keras model saved at models/{model_id}.pkl")

        min_len = min(len(actuals), len(predictions))
        actuals = actuals[:min_len]
        predictions = predictions[:min_len]

        # ✅ CRITICAL: Use the preserved original forecast_horizon
        print(f"[CRITICAL BEFORE FORECASTING] Using forecast_horizon: {original_forecast_horizon}")

        # === Future Forecasting Logic ===
        future_output = {}
        try:
            last_date = df.index[-1]
            freq = getattr(config, 'frequency', 'daily')
            
            if freq == 'daily':
                future_freq = 'D'
            elif freq == 'weekly':
                future_freq = 'W-MON'
            elif freq == 'monthly':
                future_freq = 'MS'
            else:
                future_freq = 'D'

            print(f"[FORECASTING START] Generating {original_forecast_horizon} future predictions")
            print(f"[FORECASTING] Model type: {config.modelType}, Frequency: {freq}")
            
            # Generate future dates - EXACTLY the number requested
            future_index = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=original_forecast_horizon, 
                freq=future_freq
            )
            print(f"[FORECASTING] Generated {len(future_index)} future dates")

            future_predictions = []

            if config.modelType == 'arima':
                print(f"[ARIMA] Forecasting {original_forecast_horizon} steps")
                forecast_result = model.get_forecast(steps=original_forecast_horizon)
                future_predictions = forecast_result.predicted_mean.values.tolist()
                print(f"[ARIMA] Generated {len(future_predictions)} predictions")

            elif config.modelType == 'prophet':
                print(f"[PROPHET] Forecasting {original_forecast_horizon} steps")
                future_df = pd.DataFrame({'ds': future_index})
                future_forecast = model.predict(future_df)
                if future_forecast is None or not isinstance(future_forecast, pd.DataFrame):
                    raise ValueError("[ERROR] No valid future_forecast returned from ensemble.")
                if 'yhat' not in future_forecast.columns:
                    raise ValueError("[ERROR] 'yhat' column missing in future_forecast.")
                future_predictions = future_forecast['yhat'].values.tolist()
                print(f"[PROPHET] Generated {len(future_predictions)} predictions")

            elif config.modelType == 'lstm':
                print(f"[LSTM] Forecasting {original_forecast_horizon} steps")
                sequence_length = getattr(model, 'sequence_length', 10)
                scaler = getattr(model, 'scaler', None)
                if scaler is None and hasattr(model, 'scalers'):
                    # Assume you're accessing scaler for LSTM or other internal model by name
                    scaler = model.scalers.get('lstm', None)

                # Start with the last sequence_length rows from the dataset
                future_input = df.iloc[-sequence_length:].copy()
                future_predictions = []

                for step in range(original_forecast_horizon):
                    print(f"[LSTM] Generating prediction {step + 1}/{original_forecast_horizon}")
                    
                    # Scale the input
                    scaled_input = scaler.transform(future_input)
                    X_input = scaled_input[-sequence_length:].reshape(1, sequence_length, scaled_input.shape[1])
                    
                    # Predict next value
                    next_pred_scaled = model.predict(X_input, verbose=0).flatten()[0]
                    
                    # Inverse transform to get actual value
                    dummy = np.zeros((1, scaled_input.shape[1]))
                    target_idx = df.columns.get_loc(target)
                    dummy[:, target_idx] = next_pred_scaled
                    inverse_pred = scaler.inverse_transform(dummy)[:, target_idx][0]
                    
                    future_predictions.append(float(inverse_pred))
                    
                    # Update the input sequence for next prediction
                    new_row = future_input.iloc[-1].copy()
                    new_row[target] = inverse_pred
                    future_input = pd.concat([future_input.iloc[1:], new_row.to_frame().T])

                print(f"[LSTM] Generated {len(future_predictions)} predictions")
            
            elif config.modelType == 'ets':
                print(f"[ETS] Forecasting {original_forecast_horizon} steps")
                try:
                    future_predictions = model.forecast(original_forecast_horizon)
                    if future_predictions is None or len(future_predictions) == 0:
                        raise ValueError("ETS forecast returned no predictions")
                    future_predictions = future_predictions.values.tolist()
                    print(f"[ETS] Generated {len(future_predictions)} predictions")
                except Exception as e:
                    print(f"[ETS FORECAST ERROR] {str(e)}")
                    traceback.print_exc()
                    raise ValueError(f"ETS forecasting failed: {str(e)}")
                
            elif config.modelType == 'lightgbm':
                print(f"[LIGHTGBM] Forecasting {original_forecast_horizon} steps")
                future_input = df.iloc[-1:].copy()
                future_predictions = []
                
                for step in range(original_forecast_horizon):
                    print(f"[LIGHTGBM] Generating prediction {step + 1}/{original_forecast_horizon}")
                    X_future = future_input.drop(columns=[target])
                    next_pred = model.predict(X_future.values)[0]
                    future_predictions.append(float(next_pred))
                    
                    # Update input for next iteration
                    new_row = future_input.iloc[-1].copy()
                    new_row[target] = next_pred
                    future_input = new_row.to_frame().T
                
                print(f"[LIGHTGBM] Generated {len(future_predictions)} predictions")
            elif config.ensembleLearning:
                print(f"[ENSEMBLE] Forecasting {original_forecast_horizon} steps")
                if hasattr(model, 'predict_future'):
                    future_predictions = model.predict_future(original_forecast_horizon)
                else:
                    # Fallback: iterative prediction for ensemble
                    future_input = df.iloc[-1:].copy()
                    future_predictions = []
                    
                    for step in range(original_forecast_horizon):
                        X_future = future_input.drop(columns=[target])
                        if hasattr(model, 'feature_names_'):
                            for feature in model.feature_names_:
                                if feature not in X_future.columns:
                                    X_future[feature] = 0
                            X_future = X_future[model.feature_names_]
                        
                        next_pred = model.predict(X_future)[0]
                        future_predictions.append(float(next_pred))
                        
                        # Update input for next iteration
                        new_row = future_input.iloc[-1].copy()
                        new_row[target] = next_pred
                        future_input = new_row.to_frame().T
                
                future_predictions = future_predictions[:original_forecast_horizon]
                print(f"[ENSEMBLE] Generated {len(future_predictions)} predictions")

            else:
                # Random Forest, XGBoost, etc.
                print(f"[{config.modelType.upper()}] Forecasting {original_forecast_horizon} steps")
                future_input = df.iloc[-1:].copy()
                future_predictions = []
                
                for step in range(original_forecast_horizon):
                    print(f"[{config.modelType.upper()}] Generating prediction {step + 1}/{original_forecast_horizon}")
                    
                    X_future = future_input.drop(columns=[target])
                    next_pred = model.predict(X_future.to_numpy())[0]
                    future_predictions.append(float(next_pred))
                    
                    # Simple approach: update target with prediction, keep other features same
                    new_row = future_input.iloc[-1].copy()
                    new_row[target] = next_pred
                    future_input = new_row.to_frame().T

                print(f"[{config.modelType.upper()}] Generated {len(future_predictions)} predictions")

            # ✅ FINAL VERIFICATION
            print(f"[FINAL CHECK] Requested: {original_forecast_horizon}, Generated: {len(future_predictions)}")
            print(f"[FINAL CHECK] Date range: {len(future_index)} dates")
            
            # Ensure exact match
            future_predictions = future_predictions[:original_forecast_horizon]
            future_index = future_index[:original_forecast_horizon]
            
            future_output = {
                "dates": [date.strftime('%Y-%m-%d') for date in future_index],
                "predictions": future_predictions,
                "count": len(future_predictions)  # Add count for verification
            }
            
            print(f"[SUCCESS] Returning {len(future_predictions)} future predictions")

        except Exception as forecast_error:
            print(f"[FORECAST ERROR] {forecast_error}")
            import traceback
            traceback.print_exc()
            future_output = {
                "error": str(forecast_error),
                "requested_horizon": original_forecast_horizon
            }

        response = {
            'status': 'success',
            'model_id': model_id,
            'parameters': params,
            'metrics': metrics,
            'dates': test.index.strftime('%Y-%m-%d').tolist(),
            'actual': actuals.tolist(),
            'forecasts': predictions.tolist(),
            'futureForecast': future_output,
            'requestedHorizon': original_forecast_horizon,  # Add for verification
            'dataInfo': {
                'title': target,
                'filename': 'forecast_data'
            },
            'modelInfo': {
                'type': config.modelType.upper(),
                'parameters': params,
                'features': {
                    'hyperparameterTuning': config.hyperparameterTuning,
                    'transferLearning': config.transferLearning,
                    'ensembleLearning': config.ensembleLearning
                }
            }
        }

        load_pretrained_models()
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={
            'error': str(e),
            'details': traceback.format_exc()
        })
    
@app.post("/api/export")
async def export_results(config: ExportConfig):
    try:
        results_data = config.data.get('Results', {})
        if not results_data:
            raise HTTPException(status_code=400, detail="No data provided")

        model_info = results_data.get('modelInfo', {})
       
        metrics_data = results_data.get('metrics', {})
        
        data_records = results_data.get('data', [])
     

        df = pd.DataFrame(data_records)

        metadata = {
            'Model Type': model_info.get('type', 'Unknown'),
            'Features Used': ', '.join(model_info.get('parameters', {}).get('features', [])),
            'MSE': metrics_data.get('mse', 'N/A'),
            'RMSE': metrics_data.get('rmse', 'N/A'),
            'MAE': metrics_data.get('mae', 'N/A'),
            'MAPE': metrics_data.get('mape', 'N/A')
        }
       
        if config.format == 'excel':
           
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Forecast Results', index=False, startrow=len(metadata) + 2)
                worksheet = writer.sheets['Forecast Results']
                for i, (key, value) in enumerate(metadata.items()):
                    worksheet.write(i, 0, key)
                    worksheet.write(i, 1, str(value))
            print("Excel export successful")
            output.seek(0)
            return Response(
                content=output.getvalue(),
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                headers={'Content-Disposition': 'attachment;filename=forecast_results.xlsx'}
            )
        
        elif config.format == 'json':

            output = {
                'metadata': metadata,
                'data': df.to_dict(orient='records')
            }
            print("JSON export successful")
            return JSONResponse(content=output)
        elif config.format == 'csv':
            output = df.to_csv(index=False)
            print("CSV export successful")
            return Response(
                content=output,
                media_type='text/csv',
                headers={'Content-Disposition': 'attachment;filename=forecast_results.csv'}
            
        )
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")

    except Exception as e:
        print(f"Export error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
# Move this here

if __name__ == '__main__':

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
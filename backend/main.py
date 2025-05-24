import pandas as pd
import numpy as np
import joblib
import tempfile
import os
import time
import traceback
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
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from io import BytesIO

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

def clean_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column].apply(clean_numeric_string), errors='coerce')

class DataConfig(BaseModel):
    timeColumn: str
    targetVariable: str
    frequency: str
    features: Optional[List[str]] = []
    modelType: Optional[str] = 'ensemble'
    hyperparameterTuning: Optional[bool] = False
    ensembleLearning: Optional[bool] = False
    transferLearning: Optional[bool] = False
    ensembleModels: Optional[List[str]] = ['arima', 'lstm', 'xgboost']
    ensembleMethod: Optional[str] = 'voting'
    ensembleWeights: Optional[List[float]] = None
    sourceModelId: Optional[str] = None
    order: Optional[tuple] = (1,1,1)
    seasonal_order: Optional[tuple] = None
    changepoint_prior_scale: Optional[float] = 0.05
    seasonality_prior_scale: Optional[float] = 10.0
    seasonality_mode: Optional[str] = 'additive'
    units: Optional[int] = 50
    dropout: Optional[float] = 0.2
    epochs: Optional[int] = 100
    batch_size: Optional[int] = 32
    sequence_length: Optional[int] = 10
    n_estimators: Optional[int] = 100
    max_depth: Optional[int] = 10
    learning_rate: Optional[float] = 0.1
    gamma: Optional[float] = 0
    min_child_weight: Optional[float] = 0
    subsample: Optional[float] = 0.8

class ExportConfig(BaseModel):
    format: str
    data: Dict[str, Any]

@app.get("/api/models")
async def get_available_models():
    models = []
    for model_id, model_info in trained_models.items():
        models.append({
            'id': model_id,
            'type': model_info['config'].get('modelType', 'unknown').upper(),
            'target': model_info['target'],
            'parameters': model_info['params'],
            'metrics': model_info.get('metrics', {})
        })
    return models

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
    
    df = df.groupby(config.timeColumn).agg({
        config.targetVariable: 'mean',
        **{feature: 'mean' for feature in config.features if feature in df.columns},
        **{f'{config.targetVariable}_lag{lag}': 'mean' for lag in [1, 2, 3]},
        f'{config.targetVariable}_rolling_mean': 'mean',
        f'{config.targetVariable}_rolling_std': 'mean'
    }).reset_index()
    
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
            'frequency': config.frequency
        },
        'features': config.features + [f'{config.targetVariable}_lag{lag}' for lag in [1, 2, 3]] + 
                    [f'{config.targetVariable}_rolling_mean', f'{config.targetVariable}_rolling_std']
    }
    
    return {
        "message": "Data processed successfully",
        "preview": {
            "rows": len(df),
            "start_date": df.index.min().strftime('%Y-%m-%d'),
            "end_date": df.index.max().strftime('%Y-%m-%d'),
            "columns": df.columns.tolist(),
            "sample": df.head(5).to_dict(orient='records')
        }
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
            for feature in self.feature_names_:
                if feature not in X.columns:
                    X[feature] = 0
            X = X[self.feature_names_]

        for i, model in enumerate(self.models):
            try:
                pred = None  # Safe default to avoid uninitialized use

                if isinstance(model, SARIMAXResultsWrapper):
                    steps = len(X)
                    pred = model.get_forecast(steps=steps).predicted_mean.values

                elif isinstance(model, Prophet):
                    future_dates = pd.DataFrame({'ds': X.index})
                    future_dates['ds'] = pd.to_datetime(future_dates['ds'], errors='coerce')
                    if future_dates['ds'].isnull().any():
                        raise ValueError("Invalid dates in Prophet future_dates")
                    forecast = model.predict(future_dates)
                    pred = forecast['yhat'].values

                elif isinstance(model, Model):  # LSTM
                    sequence_length = getattr(model, 'sequence_length', 10)
                    scaler = self.scalers.get(i)
                    if scaler is not None:
                        X_scaled = scaler.transform(X)
                        X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
                        sequences, _ = prepare_sequence_data(X_scaled_df, processed_data['config']['target'], sequence_length)
                        if len(sequences) == 0:
                            raise ValueError("Not enough data for LSTM prediction")
                        pred = model.predict(sequences, verbose=0).flatten()
                        pred = pred[:len(X)]
                        dummy = np.zeros((len(pred), X.shape[1]))
                        target_idx = X.columns.get_loc(processed_data['config']['target'])
                        dummy[:, target_idx] = pred
                        pred = scaler.inverse_transform(dummy)[:, target_idx]

                else:  # RandomForest, XGBoost
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

        if self.method == 'voting':
            weights = np.array([self.weights[i] for i in successful_models])
            weights = weights / weights.sum()
            return np.average(predictions, axis=0, weights=weights)
        elif self.method == 'stacking':
        # You can improve this by adding a trained meta-learner
            return predictions[0]
        else:
            return np.mean(predictions, axis=0)

        
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
    params = {
        'units': config.get('units', 50),
        'dropout': config.get('dropout', 0.2),
        'epochs': config.get('epochs', 100),
        'batch_size': config.get('batch_size', 32),
        'sequence_length': config.get('sequence_length', 10)
    }
    
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
    model = XGBRegressor(**params, n_jobs=-1)
    X_train = train.drop(columns=[processed_data['config']['target']])
    y_train = train[processed_data['config']['target']]
    model.fit(X_train.values, y_train)
    return params, model

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
    try:
        target = processed_data['config']['target']
        
        if isinstance(source_model, Model):
            input_shape = source_model.input_shape[1:]
            source_config = {layer.name: layer.get_config() for layer in source_model.layers}
            source_weights = {layer.name: layer.get_weights() for layer in source_model.layers}
            
            inputs = Input(shape=input_shape)
            x = inputs
            
            for i, layer in enumerate(source_model.layers[1:-1]):
                if isinstance(layer, (LSTM, Bidirectional)):
                    config = source_config[layer.name]
                    if isinstance(layer, Bidirectional):
                        wrapped = config['layer']['config']
                        wrapped['units'] = config.get('units', 50)
                        x = Bidirectional(LSTM(**wrapped))(x)
                    else:
                        config['units'] = config.get('units', 50)
                        x = LSTM(**config)(x)
                elif isinstance(layer, BatchNormalization):
                    x = BatchNormalization()(x)
                elif isinstance(layer, Dropout):
                    x = Dropout(config.get('dropout', 0.2))(x)
                
                if layer.name in source_weights:
                    x.layer.set_weights(source_weights[layer.name])
                    x.layer.trainable = False
            
            x = Dense(32, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(1)(x)
            
            new_model = Model(inputs=inputs, outputs=outputs)
            
            new_model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='mse',
                metrics=['mae', 'mse']
            )
            
            if hasattr(source_model, 'scaler'):
                new_model.scaler = source_model.scaler
            if hasattr(source_model, 'sequence_length'):
                new_model.sequence_length = source_model.sequence_length
                
            return new_model
            
        elif isinstance(source_model, RandomForestRegressor):
            n_estimators = min(config.get('n_estimators', 50), source_model.n_estimators)
            new_model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=source_model.max_depth,
                random_state=42,
                warm_start=True
            )
            
            if hasattr(source_model, 'estimators_'):
                new_model.estimators_ = source_model.estimators_[:n_estimators]
            
            return new_model
            
        elif isinstance(source_model, XGBRegressor):
            new_model = XGBRegressor(
                n_estimators=config.get('n_estimators', 50),
                max_depth=source_model.max_depth,
                learning_rate=config.get('learning_rate', 0.01),
                objective='reg:squarederror',
                random_state=42
            )
            
            if hasattr(source_model, 'get_booster'):
                new_model._Booster = source_model.get_booster().copy()
            
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
        'xgboost': 'XGBoost'
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
async def train_model(config: DataConfig):
    if processed_data is None:
        raise HTTPException(status_code=404, detail="No processed data available. Please process data first.")
    
    df = processed_data['data']
    target = processed_data['config']['target']
    
    split_point = int(len(df) * 0.8)
    train = df.iloc[:split_point]
    test = df.iloc[split_point:]
    
    try:
        if config.ensembleLearning:
            selected_models = config.ensembleModels
            print(selected_models)
            if config.modelType not in selected_models:
                print("Hello")
                selected_models.append(config.modelType)
            
            ensemble = create_ensemble_model(
                train, 
                test, 
                {
                    'ensembleModels': selected_models,
                    'ensembleMethod': config.ensembleMethod,
                    'ensembleWeights': config.ensembleWeights,
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
        else:
            if config.hyperparameterTuning:
                params, model = tune_hyperparameters(config.modelType, train, test, config.model_dump())
            else:
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
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown model type: {config.modelType}")

        if config.transferLearning and config.sourceModelId:
            source_model_id = config.sourceModelId
            if source_model_id in trained_models:
                source_model = trained_models[source_model_id]['model']
                source_config = trained_models[source_model_id]['config']
                source_target = trained_models[source_model_id]['target']
                
                transferred_model = apply_transfer_learning(source_model, train, config.model_dump())
                
                if transferred_model is not None:
                    if isinstance(transferred_model, Model):
                        sequence_length = getattr(source_model, 'sequence_length', 10)
                        scaler = getattr(source_model, 'scaler', MinMaxScaler())
                        transferred_model.sequence_length = sequence_length
                        transferred_model.scaler = scaler
                        
                        scaled_data = scaler.transform(train)
                        scaled_df = pd.DataFrame(scaled_data, index=train.index, columns=train.columns)
                        
                        X, y = prepare_sequence_data(
                            scaled_df,
                            target,
                            sequence_length
                        )
                        
                        for layer in transferred_model.layers[:-2]:
                            layer.trainable = False
                        
                        transferred_model.fit(
                            X, y,
                            epochs=25,
                            batch_size=32,
                            validation_split=0.2,
                            callbacks=[
                                EarlyStopping(patience=5, restore_best_weights=True),
                                ReduceLROnPlateau(factor=0.5, patience=3)
                            ],
                            verbose=0
                        )
                        
                        for layer in transferred_model.layers:
                            layer.trainable = True
                        
                        transferred_model.compile(
                            optimizer=Adam(learning_rate=0.00001),
                            loss='mse'
                        )
                        
                        transferred_model.fit(
                            X, y,
                            epochs=25,
                            batch_size=32,
                            validation_split=0.2,
                            callbacks=[
                                EarlyStopping(patience=5, restore_best_weights=True),
                                ReduceLROnPlateau(factor=0.5, patience=3)
                            ],
                            verbose=0
                        )
                    
                    else:
                        X_train = train.drop(columns=[target])
                        y_train = train[target]
                        
                        if isinstance(transferred_model, RandomForestRegressor):
                            transferred_model.warm_start = True
                            transferred_model.fit(X_train, y_train)
                        elif isinstance(transferred_model, XGBRegressor):
                            transferred_model.fit(
                                X_train, y_train,
                                xgb_model=transferred_model.get_booster(),
                                learning_rate=0.01
                            )
                    
                    model = transferred_model
                    params = {
                        'transfer_learning': True,
                        'source_model': source_model_id,
                        'source_config': source_config,
                        'source_target': source_target,
                        **config.model_dump()
                    }
                    
                    if isinstance(model, Model):
                        model.scaler = scaler
                        model.sequence_length = sequence_length
                else:
                    print("Transfer learning failed, falling back to regular training")
                    if config.modelType == 'lstm':
                        params, model = train_lstm(train, config.model_dump())
                    elif config.modelType == 'random_forest':
                        params, model = train_random_forest(train, test, config.model_dump())
                    elif config.modelType == 'xgboost':
                        params, model = train_xgboost(train, test, config.model_dump())
            
        if config.ensembleLearning:
            X_test = test.copy()
            if hasattr(model, 'feature_names_'):
                for feature in model.feature_names_:
                    if feature not in X_test.columns:
                        X_test[feature] = 0  # default fill
                X_test = X_test[model.feature_names_]  # correct order

       

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
                
                X_test, _ = prepare_sequence_data(
                    test_df,
                    target,
                    sequence_length
                )
                
                predictions = model.predict(X_test, verbose=0)
                predictions = predictions.flatten()
                
                dummy = np.zeros((len(predictions), test.shape[1]))
                target_idx = test.columns.get_loc(target)
                dummy[:, target_idx] = predictions
                predictions = scaler.inverse_transform(dummy)[:, target_idx]
                predictions = predictions[:len(test)]
            else:
                X_test = test.drop(columns=[target])
                predictions = model.predict(X_test.to_numpy())
        
        actuals = test[target].values
        metrics = evaluate_model(model, train, test, config.modelType if not config.ensembleLearning else 'ensemble')
        
        model_id = f"{config.modelType}_{int(time.time())}"
        trained_models[model_id] = {
            'model': model,
            'params': params,
            'config': config.model_dump(),
            'train_data': train,
            'test_data': test,
            'target': target,
            'metrics': metrics
        }
        
        return {
            'status': 'success',
            'model_id': model_id,
            'parameters': params,
            'metrics': metrics,
            'dates': test.index.strftime('%Y-%m-%d').tolist(),
            'actual': actuals.tolist(),
            'forecasts': predictions.tolist(),
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
        
    except Exception as e:
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
        
        if config.format in ['csv', 'excel']:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Forecast Results', index=False, startrow=len(metadata) + 2)
                worksheet = writer.sheets['ForecastResults']
                for i, (key, value) in enumerate(metadata.items()):
                    worksheet.write(i, 0, key)
                    worksheet.write(i, 1, str(value))
            
            output.seek(0)
            return Response(
                content=output.getvalue(),
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                headers={'Content-Disposition': 'attachment;filename=forecast_results.xlsx'}
            )
            
        else:  # json
            output = {
                'metadata': metadata,
                'data': df.to_dict(orient='records')
            }
            return JSONResponse(content=output)
        
    except Exception as e:
        print(f"Export error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
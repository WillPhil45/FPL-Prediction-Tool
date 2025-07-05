import pandas as pd
import numpy as np
import os
import pickle
import math
import datetime
import json
import matplotlib.pyplot as plt

# Sklearn stuff
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold, KFold

import itertools
import random

# Add Optuna for Bayesian hyperparameter optimization
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_param_importances, plot_optimization_history
from optuna.pruners import MedianPruner

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Ensure deterministic behavior for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce Optuna output

# Keras Tuner
try:
    import keras_tuner as kt
    KERAS_TUNER_AVAILABLE = True
except ImportError:
    KERAS_TUNER_AVAILABLE = False

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Check for available GPUs and print information
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Print GPU information
        print(f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        
        # Set memory growth to avoid consuming all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
        # Optionally, you can limit GPU memory if needed
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # 4GB limit
        # )
            
        print("GPU memory growth enabled")
        
        # Log that we're using GPU
        print("Training will use GPU acceleration")
        
        # Set mixed precision policy for faster training on compatible GPUs
        # This works especially well on NVIDIA GPUs with tensor cores (RTX series)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision training enabled (float16/float32)")
        
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU detected. Training will use CPU only.")

LSTM_BATCH_SIZE = 64  # Increased batch size for GPU training (adjust based on your GPU memory)
DL_BATCH_SIZE = 128   # Increased batch size for dense networks

def clear_gpu_memory():
    """Clear GPU memory between training runs to prevent memory buildup"""
    import gc
    import tensorflow as tf
    gc.collect()
    tf.keras.backend.clear_session()
    
    # For even more aggressive cleanup (if needed)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.reset_memory_stats(gpu)
            except:
                pass  # Some TF versions might not support this

# ------------------------------------------------------------------------
# 1) Load time-based train/val/test
# ------------------------------------------------------------------------
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, "..", "data")

train_path = os.path.join(data_dir, "time_train_data.csv")
val_path   = os.path.join(data_dir, "time_val_data.csv")
test_path  = os.path.join(data_dir, "time_test_data.csv")

train_df = pd.read_csv(train_path)
val_df   = pd.read_csv(val_path)
test_df  = pd.read_csv(test_path)

# Create a metadata file to store feature names for each position
metadata_dir = os.path.join(script_dir, "..", "models", "metadata")
os.makedirs(metadata_dir, exist_ok=True)

# Folder for experimental results
results_dir = os.path.join(script_dir, "..", "experimental_results")
os.makedirs(results_dir, exist_ok=True)

# Folder for Optuna study visualizations
optuna_dir = os.path.join(results_dir, "optuna_visualizations")
os.makedirs(optuna_dir, exist_ok=True)

# Timestamped results file
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_csv_path = os.path.join(results_dir, f"results_{timestamp_str}.csv")

# We'll gather results in this list of dicts, then save as CSV at the end.
experiment_results = []

# ------------------------------------------------------------------------
# 2) Positions
# ------------------------------------------------------------------------
POSITION_COLS = {
    'GK': 'position_GK',
    'DEF': 'position_DEF',
    'MID': 'position_MID',
    'FWD': 'position_FWD'
}

# ------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------

def extract_sequence_features(df):
    """
    Extract features that represent time sequences, like last_3, last_10, etc.
    Returns a dict mapping sequence length to list of base feature names
    """
    all_cols = df.columns.tolist()
    sequence_features = {}
    
    # Find columns containing _last_ in their name
    seq_indicators = ["_last_"]
    
    for col in all_cols:
        for indicator in seq_indicators:
            if indicator in col:
                # Extract base feature name and sequence length
                parts = col.split(indicator)
                if len(parts) == 2:
                    base_feature = parts[0]
                    try:
                        seq_len = int(parts[1])
                        if seq_len not in sequence_features:
                            sequence_features[seq_len] = []
                        sequence_features[seq_len].append(base_feature)
                    except ValueError:
                        continue
    
    # Remove duplicates and sort
    for seq_len in sequence_features:
        sequence_features[seq_len] = sorted(list(set(sequence_features[seq_len])))
    
    return sequence_features

def prepare_lstm_sequences(X_data, lookback=3):
    """
    Prepare data for LSTM by organizing into a 3D tensor with a lookback window.
    
    This approach treats each feature (including pre-calculated sequence features)
    as independent features, but structures them in a sequence of lookback periods.
    
    Parameters:
    - X_data: DataFrame with features, including sequence features
    - lookback: Number of time steps to include
    
    Returns:
    - X_lstm: 3D tensor with shape (samples, lookback, features)
    - feature_mapping: Dict mapping indices to feature names
    """
    # Get dimensions
    n_samples = X_data.shape[0]
    n_features = X_data.shape[1]
    
    # Initialize output array
    X_lstm = np.zeros((n_samples, lookback, n_features))
    
    # Fill the array
    for i in range(n_samples):
        # For each time step in the lookback window
        for t in range(lookback):
            # If we're looking back beyond start of data, use the earliest available data
            idx = max(0, i - t)
            # Store all features for this time step
            X_lstm[i, lookback-1-t, :] = X_data.iloc[idx].values
    
    # Create feature mapping
    feature_mapping = {i: name for i, name in enumerate(X_data.columns)}
    
    return X_lstm, feature_mapping

# ------------------------------------------------------------------------
# 3) Hyperparameter Optimization Functions with Optuna
# ------------------------------------------------------------------------

def optimize_gbr(X_train, y_train, X_val, y_val, pos_label, n_trials=30):
    """
    Optimize GradientBoostingRegressor hyperparameters using Optuna with MSE target
    """
    def objective(trial):
        # Define hyperparameter space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42
        }
        
        # Create and train model
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        
        # Calculate validation MSE
        y_val_pred = model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_val_pred)
        mae_val = mean_absolute_error(y_val, y_val_pred)
        r2_val = r2_score(y_val, y_val_pred)
        
        # Store additional metrics
        trial.set_user_attr('mae', mae_val)
        trial.set_user_attr('r2', r2_val)
        
        return mse_val
    
    # Create study with TPE sampler
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42, multivariate=True),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name=f'GBR_{pos_label}_optimization_MSE'
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=4)
    
    # Get best parameters
    best_params = study.best_params
    best_trial = study.best_trial
    
    print(f"Best GBR parameters for {pos_label}: {best_params}")
    print(f"Best metrics - MSE: {best_trial.value:.4f}, MAE: {best_trial.user_attrs['mae']:.4f}, R²: {best_trial.user_attrs['r2']:.4f}")
    
    # Visualize importance of hyperparameters
    try:
        param_importances = plot_param_importances(study)
        optimization_history = plot_optimization_history(study)
        
        # Save visualizations
        param_importances.write_image(os.path.join(optuna_dir, f"GBR_{pos_label}_param_importance_MSE.png"))
        optimization_history.write_image(os.path.join(optuna_dir, f"GBR_{pos_label}_opt_history_MSE.png"))
        print("Parameter importance and optimization history visualized and saved.")
    except Exception as e:
        print(f"Could not create visualization due to: {e}")
    
    # Create and train best model
    best_model = GradientBoostingRegressor(random_state=42, **best_params)
    best_model.fit(X_train, y_train)
    
    # Final evaluation
    y_val_pred = best_model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"Final model validation - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    return best_model, best_params

def optimize_rfr(X_train, y_train, X_val, y_val, pos_label, n_trials=30):
    """
    Optimized RandomForestRegressor hyperparameter tuning that balances speed and performance.
    Uses MSE as the target metric to minimize prediction error magnitude.
    
    Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - pos_label: Position label for logging
    - n_trials: Number of hyperparameter combinations to try
    
    Returns:
    - best_model: Trained model with best hyperparameters
    - best_params: Dictionary of best hyperparameters
    """
    print(f"Starting optimized RandomForest tuning for {pos_label} position (MSE target)...")
    
    # Pre-calculate validation set size for warm_start efficiency
    val_size = len(y_val)
    
    def objective(trial):
        # Use a more focused parameter space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'bootstrap': True,  # Fix this parameter
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error']),
            'random_state': 42,
            'n_jobs': 1,  # Critical: prevent nested parallelization
            'warm_start': True  # Enable incremental tree building
        }
        
        # First quickly evaluate with fewer trees
        quick_params = params.copy()
        quick_params['n_estimators'] = min(100, params['n_estimators'])
        
        model = RandomForestRegressor(**quick_params)
        model.fit(X_train, y_train)
        
        # Get initial validation score
        y_val_pred = model.predict(X_val)
        initial_mse = mean_squared_error(y_val, y_val_pred)
        initial_mae = mean_absolute_error(y_val, y_val_pred)
        
        # If initial performance is very poor, don't waste time on full training
        # Using a relative threshold based on data characteristics
        # For FPL points, a reasonable threshold might be around 3.0-4.0 MSE
        if initial_mse > np.var(y_val) * 1.3:  # If MSE is much worse than simple baseline
            return initial_mse
        
        # If we've passed the quick check and need more trees, add them incrementally
        if params['n_estimators'] > quick_params['n_estimators']:
            model.n_estimators = params['n_estimators']
            model.fit(X_train, y_train)
        
        # Final performance evaluation focusing on MSE
        y_val_pred = model.predict(X_val)
        final_mse = mean_squared_error(y_val, y_val_pred)
        final_mae = mean_absolute_error(y_val, y_val_pred)
        final_r2 = r2_score(y_val, y_val_pred)
        
        # Report metrics for tracking
        trial.set_user_attr('mae', final_mae)
        trial.set_user_attr('mse', final_mse)
        trial.set_user_attr('r2', final_r2)
        
        # Report progress
        if trial.number % 5 == 0:
            print(f"Trial {trial.number}: MSE={final_mse:.4f}, MAE={final_mae:.4f}, R²={final_r2:.4f}, "
                  f"n_estimators={params['n_estimators']}")
        
        return final_mse  # Return MSE directly (smaller is better)
    
    # Use TPESampler with multivariate=True for more efficient parameter sampling
    sampler = TPESampler(seed=42, multivariate=True)
    
    # Use efficient pruning
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
    
    # Create study
    study = optuna.create_study(
        direction='minimize',  # Use minimize for MSE
        sampler=sampler,
        pruner=pruner,
        study_name=f'RF_{pos_label}_optimization_MSE'
    )
    
    # Run optimization with explicit parallelization control
    study.optimize(
        objective, 
        n_trials=n_trials, 
        n_jobs=4,  # Adjust based on CPU cores
        timeout=3600,  # Add 1-hour timeout as safety
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    best_trial = study.best_trial
    
    # Report final results
    print(f"Best RF parameters for {pos_label}: {best_params}")
    print(f"Best metrics - MSE: {best_trial.user_attrs['mse']:.4f}, MAE: {best_trial.user_attrs['mae']:.4f}, R²: {best_trial.user_attrs['r2']:.4f}")
    
    # Create visualizations if possible
    try:
        param_importances = plot_param_importances(study)
        optimization_history = plot_optimization_history(study)
        
        # Save visualizations
        param_importances.write_image(os.path.join(optuna_dir, f"RF_{pos_label}_param_importance_MSE.png"))
        optimization_history.write_image(os.path.join(optuna_dir, f"RF_{pos_label}_opt_history_MSE.png"))
    except Exception as e:
        print(f"Could not create visualization due to: {e}")
    
    # Train final model with best parameters (using all cores)
    final_params = best_params.copy()
    final_params['n_jobs'] = -1  # Use all cores for final training
    final_params['warm_start'] = False  # Disable for final model
    
    best_model = RandomForestRegressor(random_state=42, **final_params)
    best_model.fit(X_train, y_train)
    
    # Evaluate final model on validation data
    y_val_pred = best_model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"Final model validation - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    return best_model, best_params

def optimize_xgb(X_train, y_train, X_val, y_val, pos_label, n_trials=30):
    """
    Optimize XGBRegressor hyperparameters using Optuna with MSE target
    """
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=50),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',  # XGBoost doesn't accept 'mse' as a valid metric name
            'random_state': 42
        }
        
        # For quick evaluation with fewer trees first
        quick_params = params.copy()
        quick_params['n_estimators'] = min(100, params['n_estimators'])
        
        # Create and train model
        model = xgb.XGBRegressor(**quick_params)
        model.fit(X_train, y_train)
        
        # Initial evaluation
        y_val_pred = model.predict(X_val)
        initial_mse = mean_squared_error(y_val, y_val_pred)
        
        # Skip full training if initial performance is poor
        if initial_mse > np.var(y_val) * 1.3:
            return initial_mse
            
        # If more trees needed, continue training
        if params['n_estimators'] > quick_params['n_estimators']:
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
        
        # Final evaluation
        y_val_pred = model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_val_pred)
        mae_val = mean_absolute_error(y_val, y_val_pred)
        r2_val = r2_score(y_val, y_val_pred)
        
        # Store additional metrics
        trial.set_user_attr('mae', mae_val)
        trial.set_user_attr('r2', r2_val)
        
        return mse_val
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42, multivariate=True),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name=f'XGB_{pos_label}_optimization_MSE'
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=4)
    
    # Get best parameters
    best_params = study.best_params
    best_trial = study.best_trial
    
    print(f"Best XGBoost parameters for {pos_label}: {best_params}")
    print(f"Best metrics - MSE: {best_trial.value:.4f}, MAE: {best_trial.user_attrs['mae']:.4f}, R²: {best_trial.user_attrs['r2']:.4f}")
    
    # Visualize importance of hyperparameters
    try:
        param_importances = plot_param_importances(study)
        optimization_history = plot_optimization_history(study)
        
        # Save visualizations
        param_importances.write_image(os.path.join(optuna_dir, f"XGB_{pos_label}_param_importance_MSE.png"))
        optimization_history.write_image(os.path.join(optuna_dir, f"XGB_{pos_label}_opt_history_MSE.png"))
    except Exception as e:
        print(f"Could not create visualization due to: {e}")
    
    # Create and train best model
    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)
    
    # Final evaluation
    y_val_pred = best_model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"Final model validation - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    return best_model, best_params

def optimize_ridge(X_train, y_train, X_val, y_val, pos_label, n_trials=20):
    """
    Optimize Ridge regression hyperparameters using Optuna with MSE target
    """
    def objective(trial):
        params = {
            'alpha': trial.suggest_float('alpha', 1e-4, 100, log=True),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'solver': trial.suggest_categorical('solver', 
                                            ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
            'random_state': 42
        }
        
        # Create and train model
        model = Ridge(**params)
        model.fit(X_train, y_train)
        
        # Calculate validation MSE
        y_val_pred = model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_val_pred)
        mae_val = mean_absolute_error(y_val, y_val_pred)
        r2_val = r2_score(y_val, y_val_pred)
        
        # Store additional metrics
        trial.set_user_attr('mae', mae_val)
        trial.set_user_attr('r2', r2_val)
        
        return mse_val
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42, multivariate=True),
        study_name=f'Ridge_{pos_label}_optimization_MSE'
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=4)
    
    # Get best parameters
    best_params = study.best_params
    best_trial = study.best_trial
    
    print(f"Best Ridge parameters for {pos_label}: {best_params}")
    print(f"Best metrics - MSE: {best_trial.value:.4f}, MAE: {best_trial.user_attrs['mae']:.4f}, R²: {best_trial.user_attrs['r2']:.4f}")
    
    # Visualize importance of hyperparameters
    try:
        param_importances = plot_param_importances(study)
        optimization_history = plot_optimization_history(study)
        
        # Save visualizations
        param_importances.write_image(os.path.join(optuna_dir, f"Ridge_{pos_label}_param_importance_MSE.png"))
        optimization_history.write_image(os.path.join(optuna_dir, f"Ridge_{pos_label}_opt_history_MSE.png"))
    except Exception as e:
        print(f"Could not create visualization due to: {e}")
    
    # Create and train best model
    best_model = Ridge(**best_params)
    best_model.fit(X_train, y_train)
    
    # Final evaluation
    y_val_pred = best_model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"Final model validation - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    return best_model, best_params

# ------------------------------------------------------------------------
# 4) Improved model builders
# ------------------------------------------------------------------------

def build_feedforward_model(input_dim, regularization=True):
    """
    Build a feedforward neural network with regularization
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    
    # First hidden layer
    if regularization:
        model.add(keras.layers.Dense(128, 
                                  kernel_regularizer=regularizers.l2(0.001),
                                  kernel_initializer='he_normal'))
    else:
        model.add(keras.layers.Dense(128, kernel_initializer='he_normal'))
    
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.Dropout(0.3))
    
    # Second hidden layer
    if regularization:
        model.add(keras.layers.Dense(64, 
                                  kernel_regularizer=regularizers.l2(0.001),
                                  kernel_initializer='he_normal'))
    else:
        model.add(keras.layers.Dense(64, kernel_initializer='he_normal'))
    
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.Dropout(0.2))
    
    # Output layer
    model.add(keras.layers.Dense(1, activation='linear'))
    
    # Compile model with Adam optimizer and learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_tunable_feedforward_model(hp, input_dim):
    """
    Build a feedforward neural network with hyperparameter tuning
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    
    # Tune number of layers
    num_layers = hp.Int('num_layers', min_value=1, max_value=4)
    
    # L2 regularization strength
    reg_strength = hp.Choice('reg_strength', [0.0, 0.0001, 0.001, 0.01])
    
    # First hidden layer
    units = hp.Int(f'units_0', min_value=32, max_value=256, step=32)
    model.add(keras.layers.Dense(
        units, 
        kernel_regularizer=regularizers.l2(reg_strength),
        kernel_initializer='he_normal'
    ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    
    dropout_rate = hp.Float(f'dropout_0', 0.0, 0.5, step=0.1)
    if dropout_rate > 0:
        model.add(keras.layers.Dropout(dropout_rate))
    
    # Additional layers
    for i in range(1, num_layers):
        units = hp.Int(f'units_{i}', min_value=16, max_value=128, step=16)
        model.add(keras.layers.Dense(
            units, 
            kernel_regularizer=regularizers.l2(reg_strength),
            kernel_initializer='he_normal'
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.1))
        
        dropout_rate = hp.Float(f'dropout_{i}', 0.0, 0.5, step=0.1)
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(keras.layers.Dense(1, activation='linear'))
    
    # Learning rate
    lr = hp.Choice('learning_rate', [1e-4, 5e-4, 1e-3, 3e-3])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_advanced_lstm_model(input_shape, regularization=True):
    """
    Build an advanced LSTM model with attention mechanisms and skip connections
    
    Parameters:
    - input_shape: Tuple (timesteps, features)
    - regularization: Whether to use regularization
    
    Returns:
    - Compiled LSTM model
    """
    # Input layer
    inputs = keras.layers.Input(shape=input_shape)
    
    # Shortcut connection (skip connection input)
    skip_connection = keras.layers.Conv1D(filters=64, kernel_size=1, padding='same')(inputs)
    skip_connection = keras.layers.BatchNormalization()(skip_connection)
    
    # First LSTM layer with return_sequences=True
    if regularization:
        lstm1 = keras.layers.Bidirectional(
            keras.layers.LSTM(
                64, 
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1,
                kernel_regularizer=regularizers.l2(0.001),
                recurrent_regularizer=regularizers.l2(0.001)
            )
        )(inputs)
    else:
        lstm1 = keras.layers.Bidirectional(
            keras.layers.LSTM(
                64,
                return_sequences=True, 
                dropout=0.2,
                recurrent_dropout=0.1
            )
        )(inputs)
    
    # Layer normalization for better stability
    lstm1 = keras.layers.LayerNormalization()(lstm1)
    
    # Self-attention mechanism
    attention_scores = keras.layers.Dense(1, activation='tanh')(lstm1)
    attention_weights = keras.layers.Softmax(axis=1)(attention_scores)
    context_vector = keras.layers.Multiply()([lstm1, attention_weights])
    
    # Second LSTM layer
    if regularization:
        lstm2 = keras.layers.Bidirectional(
            keras.layers.LSTM(
                32,
                return_sequences=True,
                dropout=0.1,
                recurrent_dropout=0.1,
                kernel_regularizer=regularizers.l2(0.001),
                recurrent_regularizer=regularizers.l2(0.001)
            )
        )(context_vector)
    else:
        lstm2 = keras.layers.Bidirectional(
            keras.layers.LSTM(
                32,
                return_sequences=True,
                dropout=0.1,
                recurrent_dropout=0.1
            )
        )(context_vector)
    
    # Apply skip connection (residual connection)
    lstm2 = keras.layers.LayerNormalization()(lstm2)
    lstm2_skip = keras.layers.Add()([lstm2, skip_connection])
    
    # Global pooling to get a fixed-size representation
    pooled = keras.layers.GlobalAveragePooling1D()(lstm2_skip)
    
    # Dense layers for final processing with residual connection
    dense1 = keras.layers.Dense(64, activation='relu')(pooled)
    dense1 = keras.layers.BatchNormalization()(dense1)
    dense1 = keras.layers.Dropout(0.2)(dense1)
    
    dense2 = keras.layers.Dense(32, activation='relu')(dense1)
    dense2 = keras.layers.BatchNormalization()(dense2)
    dense2 = keras.layers.Dropout(0.1)(dense2)
    
    # Output layer
    outputs = keras.layers.Dense(1, activation='linear')(dense2)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model with Adam optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def optimize_lstm(X_train, y_train, X_val, y_val, input_shape, pos_label, n_trials=20):
    """
    Optimize LSTM hyperparameters using Optuna with MSE target
    """
    def objective(trial):
        # Define hyperparameter space
        lstm_units1 = trial.suggest_int('lstm_units1', 32, 128)
        lstm_units2 = trial.suggest_int('lstm_units2', 16, 64)
        dense_units1 = trial.suggest_int('dense_units1', 32, 128)
        dense_units2 = trial.suggest_int('dense_units2', 16, 64)
        dropout1 = trial.suggest_float('dropout1', 0.1, 0.5)
        dropout2 = trial.suggest_float('dropout2', 0.1, 0.4)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
        
        # Create custom model
        inputs = keras.layers.Input(shape=input_shape)
        
        # IMPORTANT FIX: Skip connection must match bidirectional output size (2x units)
        # The output of Bidirectional LSTM will be 2*lstm_units1
        skip_connection = keras.layers.Conv1D(
            filters=lstm_units1 * 2,  # Multiply by 2 for bidirectional output
            kernel_size=1, 
            padding='same'
        )(inputs)
        skip_connection = keras.layers.BatchNormalization()(skip_connection)
        
        # First LSTM layer
        lstm1 = keras.layers.Bidirectional(
            keras.layers.LSTM(
                lstm_units1, 
                return_sequences=True,
                dropout=dropout1,
                recurrent_dropout=dropout1/2,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg)
            )
        )(inputs)
        
        lstm1 = keras.layers.LayerNormalization()(lstm1)
        
        # Attention mechanism
        attention_scores = keras.layers.Dense(1, activation='tanh')(lstm1)
        attention_weights = keras.layers.Softmax(axis=1)(attention_scores)
        context_vector = keras.layers.Multiply()([lstm1, attention_weights])
        
        # Second LSTM layer
        lstm2 = keras.layers.Bidirectional(
            keras.layers.LSTM(
                lstm_units2,
                return_sequences=True,
                dropout=dropout2,
                recurrent_dropout=dropout2/2,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg)
            )
        )(context_vector)
        
        lstm2 = keras.layers.LayerNormalization()(lstm2)
        
        # IMPORTANT FIX: Ensure dimensions match before adding
        # Add a projection layer to match dimensions if necessary
        if lstm_units2 * 2 != lstm_units1 * 2:
            lstm2 = keras.layers.Conv1D(
                filters=lstm_units1 * 2,
                kernel_size=1,
                padding='same'
            )(lstm2)
        
        # Apply skip connection
        lstm2_skip = keras.layers.Add()([lstm2, skip_connection])
        
        # Global pooling
        pooled = keras.layers.GlobalAveragePooling1D()(lstm2_skip)
        
        # Dense layers
        dense1 = keras.layers.Dense(dense_units1, activation='relu')(pooled)
        dense1 = keras.layers.BatchNormalization()(dense1)
        dense1 = keras.layers.Dropout(dropout1)(dense1)
        
        dense2 = keras.layers.Dense(dense_units2, activation='relu')(dense1)
        dense2 = keras.layers.BatchNormalization()(dense2)
        dense2 = keras.layers.Dropout(dropout2)(dense2)
        
        # Output layer
        outputs = keras.layers.Dense(1, activation='linear')(dense2)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model with MSE loss 
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',  # MSE loss
            metrics=['mae']
        )
        
        # Create callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=LSTM_BATCH_SIZE,  # Use GPU-optimized batch size
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Get best validation loss (MSE)
        best_val_mse = min(history.history['val_loss'])
        
        # Also calculate other metrics for the best model
        y_val_pred = model.predict(X_val, verbose=0).flatten()
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # Store metrics
        trial.set_user_attr('mae', val_mae)
        trial.set_user_attr('r2', val_r2)
        trial.set_user_attr('mse', best_val_mse)
        
        # Clean up memory
        keras.backend.clear_session()
        
        clear_gpu_memory()
        return best_val_mse
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        study_name=f'LSTM_{pos_label}_optimization_MSE'
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    best_trial = study.best_trial
    
    print(f"Best LSTM parameters for {pos_label}: {best_params}")
    print(f"Best metrics - MSE: {best_trial.value:.4f}, MAE: {best_trial.user_attrs['mae']:.4f}, R²: {best_trial.user_attrs['r2']:.4f}")
    
    # Visualize importance of hyperparameters
    try:
        param_importances = plot_param_importances(study)
        optimization_history = plot_optimization_history(study)
        
        # Save visualizations
        param_importances.write_image(os.path.join(optuna_dir, f"LSTM_{pos_label}_param_importance_MSE.png"))
        optimization_history.write_image(os.path.join(optuna_dir, f"LSTM_{pos_label}_opt_history_MSE.png"))
    except Exception as e:
        print(f"Could not create visualization due to: {e}")
    
    # Build best model
    # Extract parameters
    lstm_units1 = best_params['lstm_units1']
    lstm_units2 = best_params['lstm_units2']
    dense_units1 = best_params['dense_units1']
    dense_units2 = best_params['dense_units2']
    dropout1 = best_params['dropout1']
    dropout2 = best_params['dropout2']
    learning_rate = best_params['learning_rate']
    l2_reg = best_params['l2_reg']
    
    # Create model with best parameters
    inputs = keras.layers.Input(shape=input_shape)
    
    # FIX: Skip connection must match bidirectional output size (2x units)
    skip_connection = keras.layers.Conv1D(
        filters=lstm_units1 * 2,  # Multiply by 2 for bidirectional
        kernel_size=1, 
        padding='same'
    )(inputs)
    skip_connection = keras.layers.BatchNormalization()(skip_connection)
    
    # First LSTM layer
    lstm1 = keras.layers.Bidirectional(
        keras.layers.LSTM(
            lstm_units1, 
            return_sequences=True,
            dropout=dropout1,
            recurrent_dropout=dropout1/2,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg)
        )
    )(inputs)
    
    lstm1 = keras.layers.LayerNormalization()(lstm1)
    
    # Attention mechanism
    attention_scores = keras.layers.Dense(1, activation='tanh')(lstm1)
    attention_weights = keras.layers.Softmax(axis=1)(attention_scores)
    context_vector = keras.layers.Multiply()([lstm1, attention_weights])
    
    # Second LSTM layer
    lstm2 = keras.layers.Bidirectional(
        keras.layers.LSTM(
            lstm_units2,
            return_sequences=True,
            dropout=dropout2,
            recurrent_dropout=dropout2/2,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg)
        )
    )(context_vector)
    
    lstm2 = keras.layers.LayerNormalization()(lstm2)
    
    # FIX: Ensure dimensions match before adding by projecting lstm2 to match skip_connection
    # This is necessary because lstm_units2 * 2 may not equal lstm_units1 * 2
    lstm2 = keras.layers.Conv1D(
        filters=lstm_units1 * 2,  # Match skip connection dimension
        kernel_size=1,
        padding='same'
    )(lstm2)
    
    # Apply skip connection - now shapes should match
    lstm2_skip = keras.layers.Add()([lstm2, skip_connection])
    
    pooled = keras.layers.GlobalAveragePooling1D()(lstm2_skip)
    
    dense1 = keras.layers.Dense(dense_units1, activation='relu')(pooled)
    dense1 = keras.layers.BatchNormalization()(dense1)
    dense1 = keras.layers.Dropout(dropout1)(dense1)
    
    dense2 = keras.layers.Dense(dense_units2, activation='relu')(dense1)
    dense2 = keras.layers.BatchNormalization()(dense2)
    dense2 = keras.layers.Dropout(dropout2)(dense2)
    
    outputs = keras.layers.Dense(1, activation='linear')(dense2)
    
    best_model = keras.Model(inputs=inputs, outputs=outputs)
    
    best_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    # Get callbacks for training final model
    callbacks = get_callbacks(f"LSTM_{pos_label}_best_MSE", patience=15)
    
    # Train final model with all data
    best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=LSTM_BATCH_SIZE,  # Use GPU-optimized batch size
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    y_val_pred = best_model.predict(X_val).flatten()
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"Final model validation - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    return best_model, best_params

def optimize_dl(X_train, y_train, X_val, y_val, pos_label, n_trials=20):
    """
    Optimize feedforward Deep Learning hyperparameters using Optuna with MSE target
    """
    def objective(trial):
        # Define hyperparameter space
        n_layers = trial.suggest_int('n_layers', 2, 4)
        units_first = trial.suggest_int('units_first', 64, 256, step=32)
        dropout_first = trial.suggest_float('dropout_first', 0.1, 0.5)
        l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu'])
        batch_norm = trial.suggest_categorical('batch_norm', [True, False])
        
        # Build model
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(X_train.shape[1],)))
        
        # First hidden layer
        model.add(keras.layers.Dense(
            units_first,
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_initializer='he_normal'
        ))
        
        if batch_norm:
            model.add(keras.layers.BatchNormalization())
            
        if activation == 'relu':
            model.add(keras.layers.ReLU())
        else:
            model.add(keras.layers.LeakyReLU(negative_slope=0.1))
            
        model.add(keras.layers.Dropout(dropout_first))
        
        # Additional layers with decreasing units
        for i in range(1, n_layers):
            units = trial.suggest_int(f'units_layer_{i}', 32, units_first, step=32)
            dropout = trial.suggest_float(f'dropout_layer_{i}', 0.1, 0.4)
            
            model.add(keras.layers.Dense(
                units,
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer='he_normal'
            ))
            
            if batch_norm:
                model.add(keras.layers.BatchNormalization())
                
            if activation == 'relu':
                model.add(keras.layers.ReLU())
            else:
                model.add(keras.layers.LeakyReLU(negative_slope=0.1))
                
            model.add(keras.layers.Dropout(dropout))
        
        # Output layer
        model.add(keras.layers.Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',  # MSE loss
            metrics=['mae']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=DL_BATCH_SIZE,  # Use GPU-optimized batch size
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Get best validation loss (MSE)
        best_val_mse = min(history.history['val_loss'])
        
        # Calculate other metrics
        y_val_pred = model.predict(X_val, verbose=0).flatten()
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # Store metrics
        trial.set_user_attr('mae', val_mae)
        trial.set_user_attr('r2', val_r2)
        trial.set_user_attr('mse', best_val_mse)
        
        # Clean up
        keras.backend.clear_session()
        
        clear_gpu_memory()
        return best_val_mse
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        study_name=f'DL_{pos_label}_optimization_MSE'
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    best_trial = study.best_trial
    
    print(f"Best Deep Learning parameters for {pos_label}: {best_params}")
    print(f"Best metrics - MSE: {best_trial.value:.4f}, MAE: {best_trial.user_attrs['mae']:.4f}, R²: {best_trial.user_attrs['r2']:.4f}")
    
    # Visualize importance of hyperparameters
    try:
        param_importances = plot_param_importances(study)
        optimization_history = plot_optimization_history(study)
        
        # Save visualizations
        param_importances.write_image(os.path.join(optuna_dir, f"DL_{pos_label}_param_importance_MSE.png"))
        optimization_history.write_image(os.path.join(optuna_dir, f"DL_{pos_label}_opt_history_MSE.png"))
    except Exception as e:
        print(f"Could not create visualization due to: {e}")
    
    # Build best model
    # Extract parameters
    n_layers = best_params['n_layers']
    units_first = best_params['units_first']
    dropout_first = best_params['dropout_first']
    l2_reg = best_params['l2_reg']
    learning_rate = best_params['learning_rate']
    activation = best_params['activation']
    batch_norm = best_params['batch_norm']
    
    # Create final model with best parameters
    best_model = keras.Sequential()
    best_model.add(keras.layers.Input(shape=(X_train.shape[1],)))
    
    # First hidden layer
    best_model.add(keras.layers.Dense(
        units_first,
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_initializer='he_normal'
    ))
    
    if batch_norm:
        best_model.add(keras.layers.BatchNormalization())
        
    if activation == 'relu':
        best_model.add(keras.layers.ReLU())
    else:
        best_model.add(keras.layers.LeakyReLU(negative_slope=0.1))
        
    best_model.add(keras.layers.Dropout(dropout_first))
    
    # Additional layers
    for i in range(1, n_layers):
        units = best_params[f'units_layer_{i}']
        dropout = best_params[f'dropout_layer_{i}']
        
        best_model.add(keras.layers.Dense(
            units,
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_initializer='he_normal'
        ))
        
        if batch_norm:
            best_model.add(keras.layers.BatchNormalization())
            
        if activation == 'relu':
            best_model.add(keras.layers.ReLU())
        else:
            best_model.add(keras.layers.LeakyReLU(negative_slope=0.1))
            
        best_model.add(keras.layers.Dropout(dropout))
    
    # Output layer
    best_model.add(keras.layers.Dense(1, activation='linear'))
    
    # Compile model
    best_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    # Get callbacks for training final model
    callbacks = get_callbacks(f"DL_{pos_label}_best_MSE", patience=15)
    
    # Train final model
    best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=DL_BATCH_SIZE,  # Use GPU-optimized batch size
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    y_val_pred = best_model.predict(X_val).flatten()
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"Final model validation - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    return best_model, best_params

def get_callbacks(model_name, patience=10):
    """
    Get standard callbacks for model training
    """
    # Create model checkpoint directory
    checkpoint_dir = os.path.join(script_dir, "..", "models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when plateau is reached
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]
    
    return callbacks

# ------------------------------------------------------------------------
# Dictionaries to store final models + scalers
# ------------------------------------------------------------------------
regression_models = {}
classification_models = {}
metadata = {}

# ------------------------------------------------------------------------
# 5) Main loop over each position
# ------------------------------------------------------------------------
for pos_label, bool_col in POSITION_COLS.items():
    print(f"\n{'='*50}")
    print(f"=== Processing position: {pos_label} ===")
    print(f"{'='*50}")
    
    # Initialize metadata for this position
    metadata[pos_label] = {
        "regression_features": {},
        "classification_features": {},
        "sequence_features": {},
        "lstm_feature_mapping": {}
    }

    # Filter rows
    train_pos_df = train_df[train_df[bool_col] == 1].copy()
    val_pos_df   = val_df[val_df[bool_col] == 1].copy()
    test_pos_df  = test_df[test_df[bool_col] == 1].copy()

    # -------------- PART A: Regression (next_match_points) --------------
    drop_for_reg = [
        'season_x','team_x','name','kickoff_time','opp_team_name','game_date',
        'next_match_points', 'next_fixture', 'team', 'started_next_match','played_next_match'
    ]

    X_train_reg = train_pos_df.drop(columns=drop_for_reg, errors='ignore')
    y_train_reg = train_pos_df['next_match_points']

    X_val_reg = val_pos_df.drop(columns=drop_for_reg, errors='ignore')
    y_val_reg = val_pos_df['next_match_points']

    X_test_reg = test_pos_df.drop(columns=drop_for_reg, errors='ignore')
    y_test_reg = test_pos_df['next_match_points']
    
    # Store feature names for metadata
    metadata[pos_label]["regression_features"] = X_train_reg.columns.tolist()
    
    # Extract sequence features (for LSTM)
    sequence_features = extract_sequence_features(X_train_reg)
    metadata[pos_label]["sequence_features"] = sequence_features

    # Print shapes
    print(f"  => X_train_reg shape: {X_train_reg.shape}, y_train_reg shape: {y_train_reg.shape}")
    print(f"     y_train_reg mean={y_train_reg.mean():.2f}, std={y_train_reg.std():.2f}")
    print(f"  => X_val_reg shape:   {X_val_reg.shape}, y_val_reg shape:   {y_val_reg.shape}")
    print(f"     y_val_reg mean={y_val_reg.mean():.2f}, std={y_val_reg.std():.2f}")
    print(f"  => X_test_reg shape:  {X_test_reg.shape}, y_test_reg shape:  {y_test_reg.shape}")
    print(f"     y_test_reg mean={y_test_reg.mean():.2f}, std={y_test_reg.std():.2f}")

    # Scale
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_val_reg_scaled   = scaler_reg.transform(X_val_reg)
    X_test_reg_scaled  = scaler_reg.transform(X_test_reg)
    
    # Create DataFrame with scaled data (for easier column access)
    X_train_reg_scaled_df = pd.DataFrame(X_train_reg_scaled, columns=X_train_reg.columns)
    X_val_reg_scaled_df = pd.DataFrame(X_val_reg_scaled, columns=X_val_reg.columns)
    X_test_reg_scaled_df = pd.DataFrame(X_test_reg_scaled, columns=X_test_reg.columns)

    # ------------------ A1) GradientBoosting Regression ------------------
    print(f"\n[{pos_label}] Training GradientBoosting Regressor with Optuna...")
    
    # Use Optuna for hyperparameter optimization
    best_gb_model, best_gb_params = optimize_gbr(
        X_train_reg_scaled, y_train_reg, 
        X_val_reg_scaled, y_val_reg,
        pos_label=pos_label,
        n_trials=30  # Adjust based on computational resources
    )
    
    # Evaluate best model on test set
    y_test_pred_gb = best_gb_model.predict(X_test_reg_scaled)
    mae_gb_test = mean_absolute_error(y_test_reg, y_test_pred_gb)
    mse_gb_test = mean_squared_error(y_test_reg, y_test_pred_gb)
    rmse_gb_test = math.sqrt(mse_gb_test)
    r2_gb_test = r2_score(y_test_reg, y_test_pred_gb)
    
    print(f"GBReg Test => MAE: {mae_gb_test:.4f}, MSE: {mse_gb_test:.4f}, RMSE: {rmse_gb_test:.4f}, R^2: {r2_gb_test:.4f}")
    
    # Store model
    regression_models[(pos_label, "GBReg")] = {
        "model": best_gb_model,
        "scaler": scaler_reg,
        "params": best_gb_params
    }
    
    # Store results
    experiment_results.append({
        "timestamp": timestamp_str,
        "position": pos_label,
        "algorithm": "GradientBoosting",
        "params": str(best_gb_params),
        "mae": mae_gb_test,
        "mse": mse_gb_test,
        "rmse": rmse_gb_test,
        "r2": r2_gb_test,
        "acc": None,
        "f1": None
    })

    # ------------------ A2) RandomForest Regression ------------------
    print(f"\n[{pos_label}] Training RandomForest Regressor with Optuna...")
    
    # Use Optuna for hyperparameter optimization
    best_rf_reg_model, best_rf_params = optimize_rfr(
        X_train_reg_scaled, y_train_reg, 
        X_val_reg_scaled, y_val_reg,
        pos_label=pos_label,
        n_trials=30
    )

    # Evaluate on test
    y_test_pred_rf = best_rf_reg_model.predict(X_test_reg_scaled)
    mae_rf_test = mean_absolute_error(y_test_reg, y_test_pred_rf)
    mse_rf_test = mean_squared_error(y_test_reg, y_test_pred_rf)
    rmse_rf_test = math.sqrt(mse_rf_test)
    r2_rf_test = r2_score(y_test_reg, y_test_pred_rf)
    
    print(f"RFReg Test => MAE: {mae_rf_test:.4f}, MSE: {mse_rf_test:.4f}, RMSE: {rmse_rf_test:.4f}, R^2: {r2_rf_test:.4f}")

    # Store RF
    regression_models[(pos_label, "RFReg")] = {
        "model": best_rf_reg_model,
        "scaler": scaler_reg,
        "params": best_rf_params
    }
    
    experiment_results.append({
        "timestamp": timestamp_str,
        "position": pos_label,
        "algorithm": "RandomForest",
        "params": str(best_rf_params),
        "mae": mae_rf_test,
        "mse": mse_rf_test,
        "rmse": rmse_rf_test,
        "r2": r2_rf_test,
        "acc": None,
        "f1": None
    })

    # ------------------ A3) XGBoost Regression ------------------
    if XGBOOST_AVAILABLE:
        print(f"\n[{pos_label}] Training XGBoost Regressor with Optuna...")
        
        # Use Optuna for hyperparameter optimization
        best_xgb_model, best_xgb_params = optimize_xgb(
            X_train_reg_scaled, y_train_reg, 
            X_val_reg_scaled, y_val_reg,
            pos_label=pos_label,
            n_trials=30
        )

        # Evaluate best on test
        y_test_pred_xgb = best_xgb_model.predict(X_test_reg_scaled)
        mae_xgb_test = mean_absolute_error(y_test_reg, y_test_pred_xgb)
        mse_xgb_test = mean_squared_error(y_test_reg, y_test_pred_xgb)
        rmse_xgb_test = math.sqrt(mse_xgb_test)
        r2_xgb_test = r2_score(y_test_reg, y_test_pred_xgb)
        
        print(f"XGBoost Test => MAE: {mae_xgb_test:.4f}, MSE: {mse_xgb_test:.4f}, RMSE: {rmse_xgb_test:.4f}, R^2: {r2_xgb_test:.4f}")

        regression_models[(pos_label, "XGBoost")] = {
            "model": best_xgb_model,
            "scaler": scaler_reg,
            "params": best_xgb_params
        }
        
        experiment_results.append({
            "timestamp": timestamp_str,
            "position": pos_label,
            "algorithm": "XGBoost",
            "params": str(best_xgb_params),
            "mae": mae_xgb_test,
            "mse": mse_xgb_test,
            "rmse": rmse_xgb_test,
            "r2": r2_xgb_test,
            "acc": None,
            "f1": None
        })
    else:
        print(f"[{pos_label}] XGBoost not available; skipping...")

    # ------------------ A4) Ridge Regression ------------------
    print(f"\n[{pos_label}] Training Ridge Regression with Optuna...")
    
    # Use Optuna for hyperparameter optimization
    best_ridge_model, best_ridge_params = optimize_ridge(
        X_train_reg_scaled, y_train_reg, 
        X_val_reg_scaled, y_val_reg,
        pos_label=pos_label,
        n_trials=20
    )
    
    # Evaluate best model on test set
    y_test_pred_ridge = best_ridge_model.predict(X_test_reg_scaled)
    mae_ridge_test = mean_absolute_error(y_test_reg, y_test_pred_ridge)
    mse_ridge_test = mean_squared_error(y_test_reg, y_test_pred_ridge)
    rmse_ridge_test = math.sqrt(mse_ridge_test)
    r2_ridge_test = r2_score(y_test_reg, y_test_pred_ridge)
    
    print(f"Ridge Test => MAE: {mae_ridge_test:.4f}, MSE: {mse_ridge_test:.4f}, RMSE: {rmse_ridge_test:.4f}, R^2: {r2_ridge_test:.4f}")
    
    # Store model
    regression_models[(pos_label, "Ridge")] = {
        "model": best_ridge_model,
        "scaler": scaler_reg,
        "params": best_ridge_params
    }
    
    # Store results
    experiment_results.append({
        "timestamp": timestamp_str,
        "position": pos_label,
        "algorithm": "Ridge",
        "params": str(best_ridge_params),
        "mae": mae_ridge_test,
        "mse": mse_ridge_test,
        "rmse": rmse_ridge_test,
        "r2": r2_ridge_test,
        "acc": None,
        "f1": None
    })

    # ------------------ A5) Deep Learning with Regularization ------------------
    print(f"\n[{pos_label}] Training Deep Learning model with Optuna (MSE target)...")

    # Use Optuna for Deep Learning hyperparameter optimization
    best_dl_model, best_dl_params = optimize_dl(
        X_train_reg_scaled, y_train_reg, 
        X_val_reg_scaled, y_val_reg,
        pos_label=pos_label,
        n_trials=20
    )

    # Evaluate on test set
    y_test_pred_dl = best_dl_model.predict(X_test_reg_scaled).flatten()
    mse_dl_test = mean_squared_error(y_test_reg, y_test_pred_dl)
    rmse_dl_test = np.sqrt(mse_dl_test)
    mae_dl_test = mean_absolute_error(y_test_reg, y_test_pred_dl)
    r2_dl_test = r2_score(y_test_reg, y_test_pred_dl)

    print(f"DeepLearning Test => MSE: {mse_dl_test:.4f}, RMSE: {rmse_dl_test:.4f}, MAE: {mae_dl_test:.4f}, R²: {r2_dl_test:.4f}")

    # Store model
    regression_models[(pos_label, "DeepLearning")] = {
        "model": best_dl_model,
        "scaler": scaler_reg,
        "params": best_dl_params
    }

    # Store results
    experiment_results.append({
        "timestamp": timestamp_str,
        "position": pos_label,
        "algorithm": "DeepLearning",
        "params": str(best_dl_params),
        "mae": mae_dl_test,
        "mse": mse_dl_test,
        "rmse": rmse_dl_test,
        "r2": r2_dl_test,
        "acc": None,
        "f1": None
    })

    # ------------------ A6) Improved LSTM for Sequence Data ------------------
    print(f"\n[{pos_label}] Training improved LSTM model with attention (MSE target)...")

    # For LSTM with multiple lookback windows
    lookback_windows = [3, 5, 7, 10] 
    best_lookback = None
    best_lstm_model = None
    best_lstm_mse = float('inf')
    best_feature_mapping = None
    best_lstm_params = None  # Added to track best parameters
    best_mae_lstm_test = None
    best_r2_lstm_test = None
    best_rmse_lstm_test = None

    for lookback in lookback_windows:
        # Prepare data for this lookback window (code unchanged)
        X_lstm_train, feature_mapping = prepare_lstm_sequences(
            X_train_reg_scaled_df, 
            lookback=lookback
        )
        
        X_lstm_val, _ = prepare_lstm_sequences(
            X_val_reg_scaled_df,
            lookback=lookback
        )
        
        X_lstm_test, _ = prepare_lstm_sequences(
            X_test_reg_scaled_df,
            lookback=lookback
        )
        
        # Get input shape
        lstm_input_shape = (X_lstm_train.shape[1], X_lstm_train.shape[2])
        
        # Use MSE optimization
        lstm_model, lstm_params = optimize_lstm(
            X_lstm_train, y_train_reg,
            X_lstm_val, y_val_reg,
            lstm_input_shape,
            pos_label=pos_label,
            n_trials=15
        )
        
        # Evaluate on test set
        y_test_pred_lstm = lstm_model.predict(X_lstm_test).flatten()
        mse_lstm_test = mean_squared_error(y_test_reg, y_test_pred_lstm)
        rmse_lstm_test = np.sqrt(mse_lstm_test)
        mae_lstm_test = mean_absolute_error(y_test_reg, y_test_pred_lstm)
        r2_lstm_test = r2_score(y_test_reg, y_test_pred_lstm)
        
        print(f"Lookback {lookback} - Test metrics: MSE={mse_lstm_test:.4f}, RMSE={rmse_lstm_test:.4f}, MAE={mae_lstm_test:.4f}, R²={r2_lstm_test:.4f}")
        
        # Track best model based on MSE
        if mse_lstm_test < best_lstm_mse:
            best_lstm_mse = mse_lstm_test
            best_lstm_model = lstm_model
            best_lookback = lookback
            best_feature_mapping = feature_mapping
            best_lstm_params = lstm_params  # Save best parameters
            best_mae_lstm_test = mae_lstm_test
            best_r2_lstm_test = r2_lstm_test
            best_rmse_lstm_test = rmse_lstm_test
            
    print(f"Best LSTM found with lookback={best_lookback}, MSE={best_lstm_mse:.4f}, RMSE={best_rmse_lstm_test:.4f}, MAE={best_mae_lstm_test:.4f}, R²={best_r2_lstm_test:.4f}")

    # Save the best model to regression_models dictionary
    regression_models[(pos_label, "LSTM")] = {
        "model": best_lstm_model,
        "scaler": scaler_reg,
        "lookback": best_lookback,
        "feature_mapping": best_feature_mapping,
        "params": best_lstm_params
    }

    # Store results in experiment_results list
    experiment_results.append({
        "timestamp": timestamp_str,
        "position": pos_label,
        "algorithm": "LSTM",
        "params": f"Advanced Bidirectional LSTM with attention, lookback={best_lookback}",
        "mae": best_mae_lstm_test,
        "mse": best_lstm_mse,
        "rmse": best_rmse_lstm_test,
        "r2": best_r2_lstm_test,
        "acc": None,
        "f1": None
    })

    # -------------- PART B: Classification Models --------------
    print(f"\n[{pos_label}] Training classification models...")
    
    drop_for_class = [
        'season_x','team_x','name','kickoff_time','opp_team_name','game_date',
        'next_match_points', 'next_fixture', 'team', 'started_next_match','played_next_match'
    ]
    
    # Store feature names for metadata
    X_class = train_pos_df.drop(columns=drop_for_class, errors='ignore')
    metadata[pos_label]["classification_features"] = X_class.columns.tolist()

    # For played_next_match
    X_train_play = train_pos_df.drop(columns=drop_for_class, errors='ignore')
    y_train_play = train_pos_df['played_next_match']

    X_val_play = val_pos_df.drop(columns=drop_for_class, errors='ignore')
    y_val_play = val_pos_df['played_next_match']

    X_test_play = test_pos_df.drop(columns=drop_for_class, errors='ignore')
    y_test_play = test_pos_df['played_next_match']

    scaler_play = StandardScaler()
    X_train_play_scaled = scaler_play.fit_transform(X_train_play)
    X_val_play_scaled   = scaler_play.transform(X_val_play)
    X_test_play_scaled  = scaler_play.transform(X_test_play)

    clf_param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }
    best_clf_model = None
    best_f1 = -1
    best_clf_params = None

    all_params_clf = list(itertools.product(
        clf_param_dist['n_estimators'],
        clf_param_dist['max_depth'],
        clf_param_dist['min_samples_split'],
        clf_param_dist['min_samples_leaf'],
        clf_param_dist['class_weight']
    ))
    random.shuffle(all_params_clf)
    all_params_clf = all_params_clf[:10]

    for (n_est, md, mss, msl, cw) in all_params_clf:
        clf_temp = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=md,
            min_samples_split=mss,
            min_samples_leaf=msl,
            class_weight=cw,
            random_state=42
        )
        clf_temp.fit(X_train_play_scaled, y_train_play)
        y_val_pred = clf_temp.predict(X_val_play_scaled)
        f1_val = f1_score(y_val_play, y_val_pred, zero_division=0)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_clf_params = (n_est, md, mss, msl, cw)
            best_clf_model = clf_temp

    # Evaluate on test
    y_test_pred_play = best_clf_model.predict(X_test_play_scaled)
    acc_test_play = accuracy_score(y_test_play, y_test_pred_play)
    f1_test_play = f1_score(y_test_play, y_test_pred_play, zero_division=0)
    precision_test_play = precision_score(y_test_play, y_test_pred_play, zero_division=0)
    recall_test_play = recall_score(y_test_play, y_test_pred_play, zero_division=0)
    
    print(f"[{pos_label} | played_next] => Test ACC: {acc_test_play:.3f}, F1: {f1_test_play:.3f}, Precision: {precision_test_play:.3f}, Recall: {recall_test_play:.3f}")

    classification_models[(pos_label, "played_next", "RFClf")] = {
        "model": best_clf_model,
        "scaler": scaler_play
    }
    experiment_results.append({
        "timestamp": timestamp_str,
        "position": pos_label,
        "algorithm": "RFClf_played_next",
        "params": str(best_clf_params),
        "mae": None,
        "mse": None,
        "rmse": None,
        "r2": None,
        "acc": acc_test_play,
        "f1": f1_test_play
    })

    # For started_next_match
    X_train_start = train_pos_df.drop(columns=drop_for_class, errors='ignore')
    y_train_start = train_pos_df['started_next_match']

    X_val_start = val_pos_df.drop(columns=drop_for_class, errors='ignore')
    y_val_start = val_pos_df['started_next_match']

    X_test_start = test_pos_df.drop(columns=drop_for_class, errors='ignore')
    y_test_start = test_pos_df['started_next_match']

    scaler_start = StandardScaler()
    X_train_start_scaled = scaler_start.fit_transform(X_train_start)
    X_val_start_scaled   = scaler_start.transform(X_val_start)
    X_test_start_scaled  = scaler_start.transform(X_test_start)

    best_clf_model2 = None
    best_f1_2 = -1
    best_clf_params2 = None
    all_params_clf2 = list(itertools.product(
        clf_param_dist['n_estimators'],
        clf_param_dist['max_depth'],
        clf_param_dist['min_samples_split'],
        clf_param_dist['min_samples_leaf'],
        clf_param_dist['class_weight']
    ))
    random.shuffle(all_params_clf2)
    all_params_clf2 = all_params_clf2[:10]

    for (n_est, md, mss, msl, cw) in all_params_clf2:
        clf_temp2 = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=md,
            min_samples_split=mss,
            min_samples_leaf=msl,
            class_weight=cw,
            random_state=42
        )
        clf_temp2.fit(X_train_start_scaled, y_train_start)
        y_val_pred2 = clf_temp2.predict(X_val_start_scaled)
        f1_val2 = f1_score(y_val_start, y_val_pred2, zero_division=0)
        if f1_val2 > best_f1_2:
            best_f1_2 = f1_val2
            best_clf_params2 = (n_est, md, mss, msl, cw)
            best_clf_model2 = clf_temp2

    y_test_pred_start = best_clf_model2.predict(X_test_start_scaled)
    acc_test_start = accuracy_score(y_test_start, y_test_pred_start)
    f1_test_start = f1_score(y_test_start, y_test_pred_start, zero_division=0)
    precision_test_start = precision_score(y_test_start, y_test_pred_start, zero_division=0)
    recall_test_start = recall_score(y_test_start, y_test_pred_start, zero_division=0)
    
    print(f"[{pos_label} | started_next] => Test ACC: {acc_test_start:.3f}, F1: {f1_test_start:.3f}, Precision: {precision_test_start:.3f}, Recall: {recall_test_start:.3f}")

    classification_models[(pos_label, "started_next", "RFClf")] = {
        "model": best_clf_model2,
        "scaler": scaler_start
    }
    experiment_results.append({
        "timestamp": timestamp_str,
        "position": pos_label,
        "algorithm": "RFClf_started_next",
        "params": str(best_clf_params2),
        "mae": None,
        "mse": None,
        "rmse": None,
        "r2": None,
        "acc": acc_test_start,
        "f1": f1_test_start
    })

# ------------------------------------------------------------------------
# 6) Save models
# ------------------------------------------------------------------------
models_dir = os.path.join(script_dir, "..", "models")
os.makedirs(models_dir, exist_ok=True)

with open(os.path.join(models_dir, "regression_models.pkl"), "wb") as f:
    pickle.dump(regression_models, f)

with open(os.path.join(models_dir, "classification_models.pkl"), "wb") as f:
    pickle.dump(classification_models, f)

# Save metadata
with open(os.path.join(metadata_dir, "feature_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\nTraining complete! Models & best params + scalers saved.")

# Create model comparison table
comparison_df = pd.DataFrame(experiment_results)
comparison_table = comparison_df.pivot_table(
    index=['position', 'algorithm'],
    values=['mae', 'mse', 'rmse', 'r2', 'acc', 'f1'],
    aggfunc='mean'
)

print("\nModel Comparison Summary:")
print(comparison_table)

# Create performance visualizations
regression_df = comparison_df[comparison_df['mse'].notna()].copy()
if not regression_df.empty:
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    pivot_mse = regression_df.pivot_table(index='algorithm', columns='position', values='mse', aggfunc='mean')
    pivot_mse.plot(kind='bar', ax=plt.gca())
    plt.title('Mean Squared Error by Algorithm and Position')
    plt.ylabel('MSE (lower is better)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    pivot_r2 = regression_df.pivot_table(index='algorithm', columns='position', values='r2', aggfunc='mean')
    pivot_r2.plot(kind='bar', ax=plt.gca())
    plt.title('R² Score by Algorithm and Position')
    plt.ylabel('R² (higher is better)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"regression_performance_{timestamp_str}.png"))
    plt.close()

# ------------------------------------------------------------------------
# 7) Save experimental results as CSV
# ------------------------------------------------------------------------
results_df = pd.DataFrame(experiment_results)
results_df.to_csv(results_csv_path, index=False)
print(f"Experimental results saved to: {results_csv_path}")
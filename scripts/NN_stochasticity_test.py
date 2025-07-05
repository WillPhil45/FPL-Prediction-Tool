#!/usr/bin/env python3
"""
Standalone Neural Network Ensemble Training Script

This script trains multiple neural network models with the same hyperparameters
to account for the stochastic nature of neural network training. It loads the
best hyperparameters from a previous optimization run and trains multiple models
for each position and neural network approach.

No imports from model_training.py are required.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import pickle
import math
import datetime
import argparse
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, ReLU, Input
from tensorflow.keras.layers import LSTM, Bidirectional, LayerNormalization, Conv1D, Multiply, Softmax
from tensorflow.keras.layers import GlobalAveragePooling1D, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import re

script_dir = os.path.dirname(__file__)

# Constants for neural network approaches
NN_APPROACHES = ["DeepLearning", "LSTM"]
POSITION_COLS = {
    'GK': 'position_GK',
    'DEF': 'position_DEF',
    'MID': 'position_MID',
    'FWD': 'position_FWD'
}

def load_best_hyperparameters(results_csv_path, position, algorithm):
    """
    Load the best hyperparameters for a specific position and algorithm from results CSV.
    
    Parameters:
    - results_csv_path: Path to the CSV file with experiment results
    - position: Position label (GK, DEF, MID, FWD)
    - algorithm: Algorithm name (DeepLearning, LSTM)
    
    Returns:
    - Dictionary with best hyperparameters or None if not found
    """
    try:
        # Load the results CSV
        results_df = pd.read_csv(results_csv_path)
        
        # Filter for the specific position and algorithm
        filtered_df = results_df[(results_df['position'] == position) & 
                                (results_df['algorithm'] == algorithm)]
        
        if filtered_df.empty:
            print(f"No results found for {algorithm} and position {position}.")
            return None
        
        # Get the row with the best MSE
        best_row = filtered_df.loc[filtered_df['mse'].idxmin()]
        
        # Parse the parameters string to get a dictionary
        params_str = best_row['params']
        
        # The params are stored as a string representation of a dictionary
        # We need to convert it back to a dictionary
        if algorithm == "DeepLearning":
            # For DeepLearning, extract n_layers, units_first, etc.
            # This is hacky but works for this specific format
            params_dict = {}
            
            # Extract key-value pairs from the string
            import re
            pattern = r"'([^']+)': ([^,}]+)"
            matches = re.findall(pattern, params_str)
            
            for key, value in matches:
                # Try to convert to appropriate types
                try:
                    if value.lower() == 'true':
                        params_dict[key] = True
                    elif value.lower() == 'false':
                        params_dict[key] = False
                    elif value.lower() == 'none':
                        params_dict[key] = None
                    elif '.' in value:
                        params_dict[key] = float(value)
                    else:
                        params_dict[key] = int(value)
                except ValueError:
                    # If conversion fails, keep as string
                    params_dict[key] = value.strip("'")
            
            return params_dict
            
        elif algorithm == "LSTM":
            # For LSTM, the format is different (lookback and feature_mapping)
            lookback = None
            if "lookback=" in params_str:
                lookback_match = re.search(r"lookback=(\d+)", params_str)
                if lookback_match:
                    lookback = int(lookback_match.group(1))
            
            # Return a dictionary with the lookback
            return {"lookback": lookback or 3}  # Default to 3 if not found
            
        else:
            # Unsupported algorithm
            print(f"Unsupported algorithm: {algorithm}")
            return None
            
    except Exception as e:
        print(f"Error loading hyperparameters: {e}")
        return None

def get_callbacks(model_name, patience=10):
    """
    Get standard callbacks for model training
    """
    # Create model checkpoint directory
    checkpoint_dir = os.path.join("models", "checkpoints")
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

def build_deep_learning_model(input_dim, params):
    """
    Build a deep learning model with the given hyperparameters
    
    Parameters:
    - input_dim: Dimension of the input features
    - params: Dictionary with hyperparameters
    
    Returns:
    - Compiled model
    """
    # Extract parameters with defaults
    n_layers = params.get('n_layers', 2)
    units_first = params.get('units_first', 128)
    dropout_first = params.get('dropout_first', 0.3)
    l2_reg = params.get('l2_reg', 0.0001)
    learning_rate = params.get('learning_rate', 0.001)
    activation = params.get('activation', 'relu')
    batch_norm = params.get('batch_norm', True)
    
    # Build model
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    # First hidden layer
    model.add(Dense(
        units_first,
        kernel_regularizer=regularizers.l2(l2_reg),
        kernel_initializer='he_normal'
    ))
    
    if batch_norm:
        model.add(BatchNormalization())
        
    if activation == 'relu':
        model.add(ReLU())
    else:
        model.add(LeakyReLU(negative_slope=0.1))
        
    model.add(Dropout(dropout_first))
    
    # Additional layers
    for i in range(1, n_layers):
        # Use layer-specific parameters if available, otherwise use defaults
        units = params.get(f'units_layer_{i}', units_first // (i+1))
        dropout = params.get(f'dropout_layer_{i}', dropout_first * 0.8)
        
        model.add(Dense(
            units,
            kernel_regularizer=regularizers.l2(l2_reg),
            kernel_initializer='he_normal'
        ))
        
        if batch_norm:
            model.add(BatchNormalization())
            
        if activation == 'relu':
            model.add(ReLU())
        else:
            model.add(LeakyReLU(negative_slope=0.1))
            
        model.add(Dropout(dropout))
    
    # Output layer
    model.add(Dense(1, activation='linear'))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
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
    inputs = Input(shape=input_shape)
    
    # Shortcut connection (skip connection input)
    skip_connection = Conv1D(filters=64, kernel_size=1, padding='same')(inputs)
    skip_connection = BatchNormalization()(skip_connection)
    
    # First LSTM layer with return_sequences=True
    if regularization:
        lstm1 = Bidirectional(
            LSTM(
                64, 
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1,
                kernel_regularizer=regularizers.l2(0.001),
                recurrent_regularizer=regularizers.l2(0.001)
            )
        )(inputs)
    else:
        lstm1 = Bidirectional(
            LSTM(
                64,
                return_sequences=True, 
                dropout=0.2,
                recurrent_dropout=0.1
            )
        )(inputs)
    
    # Layer normalization for better stability
    lstm1 = LayerNormalization()(lstm1)
    
    # Self-attention mechanism
    attention_scores = Dense(1, activation='tanh')(lstm1)
    attention_weights = Softmax(axis=1)(attention_scores)
    context_vector = Multiply()([lstm1, attention_weights])
    
    # Second LSTM layer
    if regularization:
        lstm2 = Bidirectional(
            LSTM(
                32,
                return_sequences=True,
                dropout=0.1,
                recurrent_dropout=0.1,
                kernel_regularizer=regularizers.l2(0.001),
                recurrent_regularizer=regularizers.l2(0.001)
            )
        )(context_vector)
    else:
        lstm2 = Bidirectional(
            LSTM(
                32,
                return_sequences=True,
                dropout=0.1,
                recurrent_dropout=0.1
            )
        )(context_vector)
    
    # Apply skip connection (residual connection)
    lstm2 = LayerNormalization()(lstm2)
    
    # Ensure dimensions match before adding
    lstm2_projected = Conv1D(filters=64, kernel_size=1, padding='same')(lstm2)
    lstm2_skip = Add()([lstm2_projected, skip_connection])
    
    # Global pooling to get a fixed-size representation
    pooled = GlobalAveragePooling1D()(lstm2_skip)
    
    # Dense layers for final processing with residual connection
    dense1 = Dense(64, activation='relu')(pooled)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.2)(dense1)
    
    dense2 = Dense(32, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.1)(dense2)
    
    # Output layer
    outputs = Dense(1, activation='linear')(dense2)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def clear_memory():
    """Clear memory between model training runs"""
    import gc
    gc.collect()
    tf.keras.backend.clear_session()

def train_multiple_models(X_train, y_train, X_val, y_val, X_test, y_test, 
                         position, algorithm, best_params, scaler, n_models=10):
    """
    Train multiple models with the same hyperparameters to account for stochasticity.
    
    Parameters:
    - X_train, y_train, X_val, y_val, X_test, y_test: Training, validation, and test data
    - position: Position label
    - algorithm: Algorithm name (DeepLearning, LSTM)
    - best_params: Dictionary with best hyperparameters
    - scaler: Fitted StandardScaler for the features
    - n_models: Number of models to train
    
    Returns:
    - List of trained models
    - Dictionary with statistics (mean and std of metrics)
    """
    models = []
    metrics = []
    
    # Create callbacks for early stopping and learning rate reduction
    callbacks = get_callbacks(f"{algorithm}_{position}_ensemble", patience=15)
    
    # For LSTM, prepare the sequences
    if algorithm == "LSTM":
        lookback = best_params.get("lookback", 3)
        X_train_seq, feature_mapping = prepare_lstm_sequences(X_train, lookback)
        X_val_seq, _ = prepare_lstm_sequences(X_val, lookback)
        X_test_seq, _ = prepare_lstm_sequences(X_test, lookback)
    
    # Print summary of the datasets
    print(f"Training {n_models} {algorithm} models for {position}")
    print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}, Test data: {X_test.shape}")
    
    # Train multiple models
    for i in range(n_models):
        print(f"\nTraining model {i+1}/{n_models} for {position} ({algorithm})...")
        
        # Set different random seeds for each model
        seed = 42 + i
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
        
        # Build and train the model based on the algorithm
        if algorithm == "DeepLearning":
            model = build_deep_learning_model(X_train.shape[1], best_params)
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=64,  # Use larger batch size for GPU
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate on test set
            y_test_pred = model.predict(X_test).flatten()
            
        elif algorithm == "LSTM":
            # Build the LSTM model
            input_shape = (lookback, X_train.shape[1])
            model = build_advanced_lstm_model(input_shape, regularization=True)
            
            # Train the model
            history = model.fit(
                X_train_seq, y_train,
                validation_data=(X_val_seq, y_val),
                epochs=100,
                batch_size=64,  # Use larger batch size for GPU
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate on test set
            y_test_pred = model.predict(X_test_seq).flatten()
            
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        
        # Store metrics
        metrics.append({
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model_index': i
        })
        
        # Store the model
        models.append(model)
        
        print(f"Model {i+1} test metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Clear memory after each model training
        clear_memory()
    
    # Calculate statistics
    metrics_df = pd.DataFrame(metrics)
    stats = {
        'mse_mean': metrics_df['mse'].mean(),
        'mse_std': metrics_df['mse'].std(),
        'rmse_mean': metrics_df['rmse'].mean(),
        'rmse_std': metrics_df['rmse'].std(),
        'mae_mean': metrics_df['mae'].mean(),
        'mae_std': metrics_df['mae'].std(),
        'r2_mean': metrics_df['r2'].mean(),
        'r2_std': metrics_df['r2'].std(),
        'models_trained': n_models
    }
    
    print(f"\nSummary statistics for {algorithm} on {position}:")
    print(f"MSE: {stats['mse_mean']:.4f} ± {stats['mse_std']:.4f}")
    print(f"RMSE: {stats['rmse_mean']:.4f} ± {stats['rmse_std']:.4f}")
    print(f"MAE: {stats['mae_mean']:.4f} ± {stats['mae_std']:.4f}")
    print(f"R²: {stats['r2_mean']:.4f} ± {stats['r2_std']:.4f}")
    
    return models, metrics_df, stats

def find_best_model_index(metrics_df):
    """
    Find the index of the best model based on MSE
    
    Parameters:
    - metrics_df: DataFrame with metrics for all models
    
    Returns:
    - Index of the best model
    """
    return metrics_df.loc[metrics_df['mse'].idxmin()]['model_index']

def evaluate_neural_network_approaches(results_csv_path, data_dir, output_dir=None, n_models=10,
                                    positions=None, algorithms=None):
    """
    Evaluate neural network approaches by training multiple models per position.
    
    Parameters:
    - results_csv_path: Path to the CSV file with experiment results
    - data_dir: Directory with the datasets
    - output_dir: Directory to save the results (defaults to 'nn_ensemble_results')
    - n_models: Number of models to train per position and algorithm
    - positions: List of positions to process (default: all)
    - algorithms: List of algorithms to process (default: all)
    
    Returns:
    - Dictionary with evaluation results
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(results_csv_path), 'nn_ensemble_results')
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default positions and algorithms if not provided
    if positions is None:
        positions = list(POSITION_COLS.keys())
    
    if algorithms is None:
        algorithms = NN_APPROACHES
    
    # Load the training, validation, and test datasets
    train_path = os.path.join(data_dir, "time_train_data.csv")
    val_path = os.path.join(data_dir, "time_val_data.csv")
    test_path = os.path.join(data_dir, "time_test_data.csv")
    
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Loaded datasets:")
        print(f"Training: {train_df.shape}")
        print(f"Validation: {val_df.shape}")
        print(f"Test: {test_df.shape}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print(f"Please make sure the following files exist:")
        print(f"  - {train_path}")
        print(f"  - {val_path}")
        print(f"  - {test_path}")
        return None
    
    # Dictionary to store results
    all_results = {}
    
    # Train models for each position and algorithm
    for pos_label in positions:
        if pos_label not in POSITION_COLS:
            print(f"Invalid position: {pos_label}")
            continue
            
        bool_col = POSITION_COLS[pos_label]
        
        print(f"\n{'='*50}")
        print(f"=== Processing position: {pos_label} ===")
        print(f"{'='*50}")
        
        # Filter rows for this position
        train_pos_df = train_df[train_df[bool_col] == 1].copy()
        val_pos_df = val_df[val_df[bool_col] == 1].copy()
        test_pos_df = test_df[test_df[bool_col] == 1].copy()
        
        # Print dataset sizes
        print(f"Position {pos_label} dataset sizes:")
        print(f"Training: {train_pos_df.shape}")
        print(f"Validation: {val_pos_df.shape}")
        print(f"Test: {test_pos_df.shape}")
        
        # Columns to drop for regression
        drop_for_reg = [
            'season_x','team_x','name','kickoff_time','opp_team_name','game_date',
            'next_match_points', 'next_fixture', 'team', 'started_next_match','played_next_match'
        ]
        
        # Prepare data
        X_train_reg = train_pos_df.drop(columns=drop_for_reg, errors='ignore')
        y_train_reg = train_pos_df['next_match_points']
        
        X_val_reg = val_pos_df.drop(columns=drop_for_reg, errors='ignore')
        y_val_reg = val_pos_df['next_match_points']
        
        X_test_reg = test_pos_df.drop(columns=drop_for_reg, errors='ignore')
        y_test_reg = test_pos_df['next_match_points']
        
        # Scale data
        scaler_reg = StandardScaler()
        X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
        X_val_reg_scaled = scaler_reg.transform(X_val_reg)
        X_test_reg_scaled = scaler_reg.transform(X_test_reg)
        
        # Create DataFrames with scaled data
        X_train_reg_scaled_df = pd.DataFrame(X_train_reg_scaled, columns=X_train_reg.columns)
        X_val_reg_scaled_df = pd.DataFrame(X_val_reg_scaled, columns=X_val_reg.columns)
        X_test_reg_scaled_df = pd.DataFrame(X_test_reg_scaled, columns=X_test_reg.columns)
        
        # Process each neural network approach
        for algorithm in algorithms:
            if algorithm not in NN_APPROACHES:
                print(f"Invalid algorithm: {algorithm}")
                continue
                
            print(f"\n[{pos_label}] Processing {algorithm}...")
            
            # Load the best hyperparameters
            best_params = load_best_hyperparameters(results_csv_path, pos_label, algorithm)
            if best_params is None:
                print(f"Skipping {algorithm} for {pos_label} (no hyperparameters found)")
                continue
            
            print(f"Best hyperparameters for {algorithm} on {pos_label}: {best_params}")
            
            # Train multiple models
            models, metrics_df, stats = train_multiple_models(
                X_train_reg_scaled_df, y_train_reg,
                X_val_reg_scaled_df, y_val_reg,
                X_test_reg_scaled_df, y_test_reg,
                pos_label, algorithm, best_params, scaler_reg, n_models
            )
            
            # Find the best model
            best_idx = int(find_best_model_index(metrics_df))
            best_model = models[best_idx]
            
            # Save metrics
            metrics_file = os.path.join(output_dir, f"{algorithm}_{pos_label}_metrics.csv")
            metrics_df.to_csv(metrics_file, index=False)
            print(f"Saved metrics to {metrics_file}")
            
            # Save best model
            model_dir = os.path.join(output_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            
            if algorithm == "LSTM":
                # For LSTM, we need to save additional information
                lookback = best_params.get("lookback", 3)
                X_train_lstm, feature_mapping = prepare_lstm_sequences(X_train_reg_scaled_df, lookback)
                
                # Create a dictionary with model and metadata
                model_data = {
                    "model": best_model,
                    "scaler": scaler_reg,
                    "lookback": lookback,
                    "feature_mapping": feature_mapping
                }
                
                # Save the model and metadata
                model_file = os.path.join(model_dir, f"{algorithm}_{pos_label}_best_model.pkl")
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data, f)
                
            else:
                # For DeepLearning, save model and scaler
                model_data = {
                    "model": best_model,
                    "scaler": scaler_reg,
                    "params": best_params
                }
                
                # Save the model and metadata
                model_file = os.path.join(model_dir, f"{algorithm}_{pos_label}_best_model.pkl")
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data, f)
            
            print(f"Saved best model to {model_file}")
            
            # Store results
            all_results[(pos_label, algorithm)] = {
                'metrics': metrics_df,
                'stats': stats,
                'best_model_index': best_idx
            }
            
    # Create a summary CSV with stats for all models
    summary_rows = []
    for (pos_label, algorithm), results in all_results.items():
        stats = results['stats']
        row = {
            'position': pos_label,
            'algorithm': algorithm,
            'mse_mean': stats['mse_mean'],
            'mse_std': stats['mse_std'],
            'rmse_mean': stats['rmse_mean'],
            'rmse_std': stats['rmse_std'],
            'mae_mean': stats['mae_mean'],
            'mae_std': stats['mae_std'],
            'r2_mean': stats['r2_mean'],
            'r2_std': stats['r2_std'],
            'models_trained': stats['models_trained']
        }
        summary_rows.append(row)
    
    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_file = os.path.join(output_dir, "nn_ensemble_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary statistics to {summary_file}")
    
    # Create visualizations
    create_visualizations(summary_df, output_dir)
    
    return all_results

def create_visualizations(summary_df, output_dir):
    """
    Create visualizations of the results
    
    Parameters:
    - summary_df: DataFrame with summary statistics
    - output_dir: Directory to save the visualizations
    """
    # Create the plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Bar chart of MSE by position and algorithm
    plt.figure(figsize=(12, 8))
    
    # Pivot the data for plotting
    pivot_df = summary_df.pivot(index='position', columns='algorithm', values='mse_mean')
    
    # Plot the bar chart
    ax = pivot_df.plot(kind='bar', yerr=summary_df.pivot(index='position', columns='algorithm', values='mse_std'), 
                     capsize=5, rot=0, figsize=(12, 8))
    
    # Add value labels manually instead of using bar_label
    for i, position in enumerate(pivot_df.index):
        for j, algorithm in enumerate(pivot_df.columns):
            value = pivot_df.loc[position, algorithm]
            ax.text(i + (j/(len(pivot_df.columns)) - 0.5 + 1/(2*len(pivot_df.columns))), 
                   value + 0.05, 
                   f'{value:.3f}', 
                   ha='center', va='bottom', fontsize=8)
    
    plt.title('Mean Squared Error by Position and Algorithm', fontsize=14)
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.legend(title='Algorithm', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "mse_by_position_algorithm.png"), dpi=300)
    plt.close()
    
    # Bar chart of R2 by position and algorithm
    plt.figure(figsize=(12, 8))
    
    # Pivot the data for plotting
    pivot_df = summary_df.pivot(index='position', columns='algorithm', values='r2_mean')
    
    # Plot the bar chart
    ax = pivot_df.plot(kind='bar', yerr=summary_df.pivot(index='position', columns='algorithm', values='r2_std'), 
                      capsize=5, rot=0, figsize=(12, 8))
    
        # Add value labels manually instead of using bar_label
    for i, position in enumerate(pivot_df.index):
        for j, algorithm in enumerate(pivot_df.columns):
            value = pivot_df.loc[position, algorithm]
            ax.text(i + (j/(len(pivot_df.columns)) - 0.5 + 1/(2*len(pivot_df.columns))), 
                   value + 0.05, 
                   f'{value:.3f}', 
                   ha='center', va='bottom', fontsize=8)

    plt.title('R² Score by Position and Algorithm', fontsize=14)
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('R²', fontsize=12)
    plt.legend(title='Algorithm', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "r2_by_position_algorithm.png"), dpi=300)
    plt.close()
    
    # Performance variability visualization - box plot approximation
    plt.figure(figsize=(14, 10))
    
    # Plot MSE variability as box plots
    positions = summary_df['position'].unique()
    algorithms = summary_df['algorithm'].unique()
    
    # Create box plot data
    boxplot_data = []
    labels = []
    
    for algorithm in algorithms:
        for position in positions:
            row = summary_df[(summary_df['position'] == position) & 
                           (summary_df['algorithm'] == algorithm)]
            if not row.empty:
                mean = row['mse_mean'].values[0]
                std = row['mse_std'].values[0]
                
                # Generate a normal distribution around the mean with the given std
                # This is a visualization approximation since we don't have the raw data
                data = np.random.normal(mean, std, 100)
                
                boxplot_data.append(data)
                labels.append(f"{position} - {algorithm}")
    
    # Create the box plot
    plt.boxplot(boxplot_data, labels=labels, showfliers=False)
    plt.title('MSE Variability by Position and Algorithm', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('MSE', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "mse_variability.png"), dpi=300)
    plt.close()
    
    print(f"Saved visualizations to {plots_dir}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train multiple neural network models to account for stochasticity')
    
    parser.add_argument('--results-csv', type=str, required=True,
                      help='Path to the CSV file with hyperparameter tuning results')
    
    parser.add_argument('--data-dir', type=str, default=os.path.join(script_dir, '..', 'data'),
                      help='Directory with the datasets')
    
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Directory to save the results (defaults to nn_ensemble_results)')
    
    parser.add_argument('--n-models', type=int, default=30,
                      help='Number of models to train per position and algorithm')
    
    parser.add_argument('--positions', type=str, nargs='+', default=None,
                      choices=['GK', 'DEF', 'MID', 'FWD'],
                      help='Positions to process (default: all)')
    
    parser.add_argument('--algorithms', type=str, nargs='+', default=None,
                      choices=['DeepLearning', 'LSTM'],
                      help='Neural network algorithms to process (default: all)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Run the evaluation
    evaluate_neural_network_approaches(
        results_csv_path=args.results_csv,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_models=args.n_models,
        positions=args.positions,
        algorithms=args.algorithms
    )
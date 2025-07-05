import os
import pickle
import random
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import shap
import lime
import lime.lime_tabular
from typing import Union, List, Tuple, Dict, Any, Optional
import glob
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def get_position_label(row):
    """Return 'FWD','MID','DEF','GK' based on the row's boolean columns."""
    if row.get('position_FWD', 0) == 1:
        return 'FWD'
    elif row.get('position_MID', 0) == 1:
        return 'MID'
    elif row.get('position_DEF', 0) == 1:
        return 'DEF'
    elif row.get('position_GK', 0) == 1:
        return 'GK'
    else:
        return 'UNK'

POSITION_ORDER = {'FWD': 0, 'MID': 1, 'DEF': 2, 'GK': 3}

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate predictions for one or more gameweeks')
    
    # Gameweek selection arguments
    gw_group = parser.add_mutually_exclusive_group()
    gw_group.add_argument('--gw', type=int, help='Specific gameweek to process')
    gw_group.add_argument('--gw-range', type=str, help='Range of gameweeks to process (e.g., "1-10")')
    gw_group.add_argument('--all-gws', action='store_true', help='Process all available gameweeks')
    
    # Explainability options
    parser.add_argument('--skip-explain', action='store_true', help='Skip generating explainability figures')
    parser.add_argument('--explain-global', action='store_true', help='Generate only global explainability figures')
    parser.add_argument('--explain-local', action='store_true', help='Generate only local explainability figures')
    parser.add_argument('--explain-method', choices=['shap', 'lime', 'both'], default='shap', 
                      help='Explainability method to use (default: shap)')
    parser.add_argument('--max-features', type=int, default=10, 
                      help='Maximum number of features to display in explainability charts')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default=None, 
                      help='Custom output directory for predictions')
    parser.add_argument('--include-all-players', action='store_true',
                      help='Include players without fixtures in predictions')
    parser.add_argument('--create-summary', action='store_true',
                      help='Create summary files combining all processed gameweeks')
    
    # Performance options
    parser.add_argument('--parallel', action='store_true',
                      help='Process gameweeks in parallel (one process per gameweek)')
    parser.add_argument('--max-processes', type=int, default=None,
                      help='Maximum number of parallel processes to use')
    
    return parser.parse_args()


# ------------------- Explainability Functions -------------------

def create_explanation_directories(output_dir: str, algorithm: str) -> Tuple[str, str, str]:
    """Create necessary directories for explainability outputs"""
    algo_dir = os.path.join(output_dir, algorithm)
    local_dir = os.path.join(algo_dir, "local_explanations")
    global_dir = os.path.join(algo_dir, "global_explanations")
    
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(global_dir, exist_ok=True)
    
    return algo_dir, local_dir, global_dir

def get_shap_explainer(model, X_background, model_type: str):
    """Get the appropriate SHAP explainer for the model type"""
    # Sample a smaller dataset for background if needed
    if hasattr(X_background, 'shape') and X_background.shape[0] > 100:
        np.random.seed(42)
        background_indices = np.random.choice(X_background.shape[0], 100, replace=False)
        background = X_background[background_indices] if isinstance(X_background, np.ndarray) else X_background.iloc[background_indices]
    else:
        background = X_background
    
    # Choose explainer based on model type
    try:
        if model_type in ['RFReg', 'GBReg', 'XGBoost', 'RFClf'] or 'RandomForest' in str(type(model)) or 'GradientBoosting' in str(type(model)):
            return shap.TreeExplainer(model)
        
        elif model_type in ['Ridge', 'LinearRegression'] or 'linear' in str(type(model)).lower():
            return shap.LinearExplainer(model, background)
        
        elif model_type in ['DeepLearning', 'LSTM'] or 'keras' in str(type(model)).lower():
            # For neural networks, use DeepExplainer directly
            try:
                # For LSTM models, reshape the background data appropriately
                if model_type == 'LSTM' or 'lstm' in str(type(model)).lower():
                    # Reshape for LSTM (samples, timesteps, features)
                    if isinstance(background, np.ndarray) and background.ndim == 2:
                        # Convert background from 2D to 3D with timesteps
                        lookback = 3  # Default lookback window size
                        background_lstm = np.zeros((background.shape[0], lookback, background.shape[1]))
                        for i in range(background.shape[0]):
                            for t in range(lookback):
                                background_lstm[i, t, :] = background[i]
                        background = background_lstm
                        print(f"Reshaped background data for LSTM: {background.shape}")
                    
                # Create DeepExplainer with appropriately shaped background data
                print(f"Creating DeepExplainer for {model_type} model")
                explainer = shap.DeepExplainer(model, background)
                return explainer
            
            except Exception as e:
                print(f"DeepExplainer failed: {e}. Trying GradientExplainer...")
                try:
                    # Try GradientExplainer as fallback
                    explainer = shap.GradientExplainer(model, background)
                    return explainer
                except Exception as e2:
                    print(f"GradientExplainer failed: {e2}. Falling back to KernelExplainer...")
                    # Last resort: use KernelExplainer
                    predict_fn = lambda x: model.predict(x).flatten()
                    return shap.KernelExplainer(predict_fn, background)
        
        else:
            # Fallback to KernelExplainer for any other model
            predict_fn = lambda x: model.predict(x).flatten()
            return shap.KernelExplainer(predict_fn, background)
    
    except Exception as e:
        print(f"Error creating SHAP explainer: {e}. Using KernelExplainer as fallback.")
        # Last resort fallback
        try:
            predict_fn = lambda x: model.predict(x).flatten()
            return shap.KernelExplainer(predict_fn, np.zeros((1, background.shape[1])) if hasattr(background, 'shape') else np.zeros((1, len(background))))
        except:
            print("Could not create any SHAP explainer. Skipping SHAP explanations.")
            return None

def get_lime_explainer(X_train, feature_names, categorical_features=None):
    """Create a LIME explainer for tabular data"""
    if categorical_features is None:
        categorical_features = []
    
    try:
        return lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            categorical_features=categorical_features,
            discretize_continuous=True
        )
    except Exception as e:
        print(f"Error creating LIME explainer: {e}")
        return None

def generate_global_shap_explanations(model, X, feature_names, model_type, output_path, position, max_features=10):
    """Generate global SHAP explanations and visualizations"""
    print(f"Generating global SHAP explanations for {model_type} - {position}...")
    
    try:
        # Sample data if it's too large
        if hasattr(X, 'shape') and X.shape[0] > 100:
            np.random.seed(42)
            indices = np.random.choice(X.shape[0], 100, replace=False)
            X_sample = X[indices] if isinstance(X, np.ndarray) else X.iloc[indices]
        else:
            X_sample = X
        
        # Special handling for LSTM models
        if model_type == 'LSTM':
            # Convert the sample data to LSTM format (samples, timesteps, features)
            lookback = 3  # Default lookback window size
            X_lstm = np.zeros((X_sample.shape[0], lookback, X_sample.shape[1]))
            for i in range(X_sample.shape[0]):
                for t in range(lookback):
                    X_lstm[i, t, :] = X_sample[i]
            
            # Get the SHAP explainer (DeepExplainer will be used from get_shap_explainer)
            explainer = get_shap_explainer(model, X_lstm, model_type)
            
            # Calculate SHAP values using the formatted data
            if explainer is not None:
                # Ensure we're using the LSTM-formatted data for SHAP calculations
                shap_values = explainer.shap_values(X_lstm)
                
                # Average across timesteps for visualization if needed
                if isinstance(shap_values, list):
                    # Handle multi-output case
                    shap_values = shap_values[0]
                
                # If we have a 4D array (samples, timesteps, features, outputs) or 3D (samples, timesteps, features)
                # Average across timesteps to get a 2D array for visualization
                if shap_values.ndim > 2:
                    shap_values = np.mean(shap_values, axis=1)  # Average across timesteps
                    print(f"Averaged SHAP values across timesteps: {shap_values.shape}")
                
                # Get base value
                if hasattr(explainer, 'expected_value'):
                    if isinstance(explainer.expected_value, list):
                        base_value = explainer.expected_value[0]
                    else:
                        base_value = explainer.expected_value
                else:
                    base_value = 0
            else:
                print("No explainer available for LSTM, skipping SHAP explanations.")
                return
                
        elif model_type == 'DeepLearning' or 'keras' in str(type(model)).lower():
            # For standard deep learning models, use DeepExplainer directly
            explainer = get_shap_explainer(model, X_sample, model_type)
            
            if explainer is not None:
                # Get SHAP values
                shap_values = explainer.shap_values(X_sample)
                
                # Handle multi-output case
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                # Get base value
                if hasattr(explainer, 'expected_value'):
                    if isinstance(explainer.expected_value, list):
                        base_value = explainer.expected_value[0]
                    else:
                        base_value = explainer.expected_value
                else:
                    base_value = 0
            else:
                print("No explainer available for DeepLearning, skipping SHAP explanations.")
                return
        else:
            # Standard models (non-neural networks)
            explainer = get_shap_explainer(model, X_sample, model_type)
            
            if explainer is None:
                print(f"No explainer available for {model_type}, skipping SHAP explanations.")
                return
                
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different formats of SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For models with multi-output
        
        # Create beeswarm plot
        plt.figure(figsize=(12, 8))
        
        # For visualization, we need 2D data
        visualization_data = X_sample
        if isinstance(X_sample, np.ndarray) and X_sample.ndim > 2:
            # If we have 3D data (like LSTM), use the flattened 2D version
            visualization_data = X_sample[:, 0, :]
        
        shap.summary_plot(
            shap_values, 
            visualization_data if isinstance(visualization_data, np.ndarray) else visualization_data.values, 
            feature_names=feature_names,
            max_display=max_features,
            show=False
        )
        plt.title(f"{model_type} Global Feature Importance - {position}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"global_beeswarm_{position}.png"), dpi=150)
        plt.close()
        
        # Create bar plot of mean absolute SHAP values
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            visualization_data if isinstance(visualization_data, np.ndarray) else visualization_data.values,
            feature_names=feature_names,
            max_display=max_features,
            plot_type="bar",
            show=False
        )
        plt.title(f"{model_type} Mean Feature Impact - {position}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"global_bar_{position}.png"), dpi=150)
        plt.close()
        
        print(f"  -> Global SHAP explanations saved to {output_path}")
        
    except Exception as e:
        print(f"Error generating global SHAP explanations: {e}")
        import traceback
        traceback.print_exc()

def generate_global_lime_explanations(model, X, feature_names, model_type, output_path, position, max_features=10):
    """Generate global LIME feature importance"""
    print(f"Generating global LIME explanations for {model_type} - {position}...")
    
    try:
        # Sample data if it's too large
        if hasattr(X, 'shape') and X.shape[0] > 100:
            np.random.seed(42)
            indices = np.random.choice(X.shape[0], 100, replace=False)
            X_sample = X[indices] if isinstance(X, np.ndarray) else X.iloc[indices]
        else:
            X_sample = X
        
        # Convert to numpy array if needed
        if not isinstance(X_sample, np.ndarray):
            X_sample = X_sample.values
        
        # Create LIME explainer
        lime_explainer = get_lime_explainer(X_sample, feature_names)
        if lime_explainer is None:
            return
        
        # Special handling for LSTM
        if model_type == 'LSTM':
            def predict_fn(x):
                # Reshape for LSTM
                x_3d = np.zeros((1, 3, x.shape[0]))  # Using lookback=3 as default
                for t in range(3):
                    x_3d[0, t, :] = x
                return model.predict(x_3d).flatten()
        else:
            # Standard prediction function
            predict_fn = lambda x: model.predict(x.reshape(1, -1)).flatten()
        
        # Get feature importances across multiple samples
        feature_importances = np.zeros(len(feature_names))
        num_samples = min(20, X_sample.shape[0])  # Limit to 20 samples for efficiency
        
        for i in range(num_samples):
            try:
                explanation = lime_explainer.explain_instance(
                    X_sample[i], 
                    predict_fn, 
                    num_features=min(len(feature_names), max_features)
                )
                
                # Extract feature importances
                for feature, importance in explanation.as_list():
                    # Extract feature name from string (LIME may add conditions)
                    feature_name = feature.split(" ")[0]
                    if feature_name in feature_names:
                        idx = feature_names.index(feature_name)
                        feature_importances[idx] += abs(importance)
            except Exception as e:
                print(f"Error generating LIME explanation for sample {i}: {e}")
                continue
        
        # Average and normalize
        feature_importances /= max(1, num_samples)
        
        # Sort features by importance
        indices = np.argsort(feature_importances)[::-1][:max_features]
        top_features = [feature_names[i] for i in indices]
        top_importances = [feature_importances[i] for i in indices]
        
        # Plot bar chart
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_importances, align='center')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Average Absolute Importance')
        plt.title(f"{model_type} Global LIME Feature Importance - {position}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"lime_global_importance_{position}.png"), dpi=150)
        plt.close()
        
        print(f"  -> Global LIME explanations saved to {output_path}")
    except Exception as e:
        print(f"Error generating global LIME explanations: {e}")

def generate_local_shap_explanation(model, x, X_background, feature_names, model_type, 
                                   output_path, position, player_name, predicted_value, max_features=10, 
                                   original_data=None):
    """Generate local SHAP explanation (waterfall plot) for a single prediction"""
    print(f"Generating local SHAP explanation for {player_name} ({position})...")
    
    try:
        # Ensure x is the right shape - always make it 2D with a single row if it's 1D
        if x.ndim == 1:
            x_reshaped = x.reshape(1, -1)
        else:
            x_reshaped = x
            
        # Store original unscaled data for display if provided
        unscaled_data = original_data if original_data is not None else x_reshaped
        
        # Special handling for LSTM models
        if model_type == 'LSTM':
            # Reshape input data for LSTM
            lookback = 3  # Default lookback
            x_lstm = np.zeros((1, lookback, x_reshaped.shape[1]))
            for t in range(lookback):
                x_lstm[0, t, :] = x_reshaped[0]
            
            # Also reshape background data if needed
            if X_background is not None and X_background.ndim == 2:
                bg_lstm = np.zeros((min(100, X_background.shape[0]), lookback, X_background.shape[1]))
                for i in range(bg_lstm.shape[0]):
                    for t in range(lookback):
                        bg_lstm[i, t, :] = X_background[i % X_background.shape[0]]
                
                # Create DeepExplainer with properly formatted background data
                try:
                    explainer = shap.DeepExplainer(model, bg_lstm)
                    # Calculate SHAP values
                    shap_values = explainer.shap_values(x_lstm)
                    
                    # For visualization, we need to reduce the 3D values to 2D
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]  # Get the first output for multi-output models
                    
                    # Average across timesteps
                    if shap_values.ndim > 2:
                        shap_values = np.mean(shap_values, axis=1)[0]  # Get the single sample and average timesteps
                    elif shap_values.ndim == 2:
                        shap_values = shap_values[0]  # Just get the single sample
                    
                    # Get the base value
                    if hasattr(explainer, 'expected_value'):
                        if isinstance(explainer.expected_value, list):
                            base_value = explainer.expected_value[0]
                        else:
                            base_value = explainer.expected_value
                    else:
                        base_value = predicted_value * 0.5
                        
                    print(f"LSTM DeepExplainer successful, shap_values shape: {shap_values.shape}")
                except Exception as e:
                    print(f"DeepExplainer failed for LSTM: {e}")
                    # Fall back to KernelExplainer
                    try:
                        predict_fn = lambda x: model.predict(x.reshape(1, lookback, -1)).flatten()[0]
                        explainer = shap.KernelExplainer(predict_fn, X_background[:50])
                        shap_values = explainer.shap_values(x_reshaped[0])
                        base_value = explainer.expected_value
                    except Exception as e2:
                        print(f"KernelExplainer also failed: {e2}")
                        raise e2
            else:
                # If we don't have background data, use a minimal fallback
                # This is less accurate but better than nothing
                print("No background data for LSTM, using minimal background")
                bg_minimal = np.zeros((2, lookback, x_reshaped.shape[1]))
                bg_minimal[1] = x_lstm[0]  # Use the sample itself as one background point
                
                try:
                    explainer = shap.DeepExplainer(model, bg_minimal)
                    shap_values = explainer.shap_values(x_lstm)
                    
                    # Process SHAP values as above
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    
                    if shap_values.ndim > 2:
                        shap_values = np.mean(shap_values, axis=1)[0]
                    elif shap_values.ndim == 2:
                        shap_values = shap_values[0]
                    
                    if hasattr(explainer, 'expected_value'):
                        if isinstance(explainer.expected_value, list):
                            base_value = explainer.expected_value[0]
                        else:
                            base_value = explainer.expected_value
                    else:
                        base_value = predicted_value * 0.5
                except Exception as e:
                    print(f"Minimal background DeepExplainer failed: {e}")
                    raise e
                    
        # Handle standard neural network models
        elif model_type == 'DeepLearning' or 'keras' in str(type(model)).lower():
            # Create background sample for DeepExplainer
            if X_background is not None:
                bg_sample = X_background[:min(100, X_background.shape[0])]
                
                try:
                    explainer = shap.DeepExplainer(model, bg_sample)
                    shap_values = explainer.shap_values(x_reshaped)
                    
                    # Process SHAP values
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    
                    if shap_values.ndim > 1 and shap_values.shape[0] == 1:
                        shap_values = shap_values[0]
                    
                    if hasattr(explainer, 'expected_value'):
                        if isinstance(explainer.expected_value, list):
                            base_value = explainer.expected_value[0]
                        else:
                            base_value = explainer.expected_value
                    else:
                        base_value = predicted_value * 0.5
                    
                    print(f"DeepLearning DeepExplainer successful, shap_values shape: {shap_values.shape}")
                except Exception as e:
                    print(f"DeepExplainer failed for DeepLearning: {e}")
                    # Fall back to GradientExplainer
                    try:
                        explainer = shap.GradientExplainer(model, bg_sample)
                        shap_values = explainer.shap_values(x_reshaped)[0]
                        base_value = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value
                    except Exception as e2:
                        print(f"GradientExplainer also failed: {e2}")
                        # Final fallback to KernelExplainer
                        predict_fn = lambda x: model.predict(x).flatten()[0]
                        explainer = shap.KernelExplainer(predict_fn, X_background[:50])
                        shap_values = explainer.shap_values(x_reshaped[0])
                        base_value = explainer.expected_value
            else:
                # If no background data, use a minimal approach
                print("No background data for DeepLearning, using KernelExplainer")
                predict_fn = lambda x: model.predict(x.reshape(1, -1)).flatten()[0]
                explainer = shap.KernelExplainer(predict_fn, np.zeros((1, x_reshaped.shape[1])))
                shap_values = explainer.shap_values(x_reshaped[0])
                base_value = explainer.expected_value
                
        # Classification models (handle differently)
        elif 'clf' in model_type.lower() or hasattr(model, 'predict_proba'):
            # For classification, use TreeExplainer for tree-based models
            if hasattr(model, 'feature_importances_') or 'rf' in model_type.lower() or 'gbm' in model_type.lower():
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(x_reshaped)
                
                # Extract positive class values
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    shap_values = shap_values[1]  # Class 1 values
                
                if hasattr(explainer, 'expected_value'):
                    if isinstance(explainer.expected_value, list) and len(explainer.expected_value) > 1:
                        base_value = explainer.expected_value[1]
                    else:
                        base_value = explainer.expected_value
                else:
                    base_value = 0.5
            else:
                # For non-tree classification models, use KernelExplainer
                if X_background is not None:
                    bg_sample = X_background[:min(50, X_background.shape[0])]
                    
                    try:
                        predict_fn = lambda x: model.predict_proba(x.reshape(-1, x_reshaped.shape[1]))[:, 1]
                        explainer = shap.KernelExplainer(predict_fn, bg_sample)
                        shap_values = explainer.shap_values(x_reshaped[0])
                        base_value = explainer.expected_value
                    except Exception as e:
                        print(f"KernelExplainer failed for classification: {e}")
                        raise e
                else:
                    print("No background data for classification model, using minimal approach")
                    predict_fn = lambda x: model.predict_proba(x.reshape(1, -1))[:, 1]
                    explainer = shap.KernelExplainer(predict_fn, np.zeros((1, x_reshaped.shape[1])))
                    shap_values = explainer.shap_values(x_reshaped[0])
                    base_value = explainer.expected_value
        
        # Standard regression models
        else:
            explainer = get_shap_explainer(model, X_background, model_type)
            if explainer is None:
                print(f"Could not create explainer for {model_type}, skipping")
                return
                
            # Calculate SHAP values
            shap_values = explainer.shap_values(x_reshaped)
            
            # Get the base value (expected value)
            if hasattr(explainer, 'expected_value'):
                base_value = explainer.expected_value
                if isinstance(base_value, list):
                    base_value = base_value[0]
            else:
                # If no expected_value, use mean of prediction as base
                base_value = predicted_value * 0.5
        
        # Handle different formats of SHAP values
        if isinstance(shap_values, list) and not isinstance(shap_values[0], np.ndarray):
            shap_values = np.array(shap_values)
        
        # Ensure we have a 1D array of values
        if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1:
            if shap_values.shape[0] == 1:
                shap_values = shap_values[0]
        
        # Ensure shap_values has the right length - should match feature_names
        if len(shap_values) != len(feature_names):
            print(f"Warning: SHAP values length ({len(shap_values)}) doesn't match feature names length ({len(feature_names)})")
            # Truncate or pad shap_values to match feature_names length
            if len(shap_values) > len(feature_names):
                shap_values = shap_values[:len(feature_names)]
            else:
                # Pad with zeros
                shap_values = np.pad(shap_values, (0, len(feature_names) - len(shap_values)))
        
        # Get unscaled data for display
        if original_data is not None:
            if hasattr(original_data, 'iloc'):
                # If we have a pandas DataFrame/Series with .iloc
                display_data = original_data.values if hasattr(original_data, 'values') else original_data
            elif hasattr(original_data, 'values'):
                # If we have a pandas object but without using .iloc
                display_data = original_data.values
            else:
                # Use the data directly
                display_data = original_data
        else:
            # No original data provided, use the input data
            display_data = x_reshaped[0] if x_reshaped.ndim > 1 else x_reshaped
        
        # Ensure display_data is 1D and same length as shap_values
        if hasattr(display_data, 'flatten'):
            display_data = display_data.flatten()
        
        if len(display_data) != len(shap_values):
            print(f"Warning: Display data length ({len(display_data)}) doesn't match SHAP values length ({len(shap_values)})")
            # Truncate or pad display_data to match shap_values length
            if len(display_data) > len(shap_values):
                display_data = display_data[:len(shap_values)]
            else:
                # Pad with zeros
                display_data = np.pad(display_data, (0, len(shap_values) - len(display_data)))
        
        # Create SHAP explanation object
        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=display_data,  # Use display data for visualization
            feature_names=feature_names
        )
        
        # Generate waterfall plot showing how features contribute to the prediction
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            explanation, 
            max_display=max_features, 
            show=False
        )
        
        plt.title(f"{model_type} Prediction Explanation\n{player_name} ({position})\nPredicted Points: {predicted_value:.2f}")
        plt.tight_layout()
        
        # Clean player name for filename
        clean_name = player_name.replace('/', '_').replace(' ', '_').replace("'", "")
        plt.savefig(os.path.join(output_path, f"local_waterfall_{position}_{clean_name}.png"), dpi=150)
        plt.close()
        
        # Also try to create a force plot for additional insight
        try:
            plt.figure(figsize=(12, 4))
            shap.plots.force(
                explanation,
                matplotlib=True,
                show=False
            )
            plt.title(f"{model_type} Force Plot - {player_name} ({position})")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"local_force_{position}_{clean_name}.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Could not generate force plot: {e}")
        
        print(f"  -> Local SHAP explanation saved to {output_path}")
    except Exception as e:
        print(f"Error generating local SHAP explanation: {e}")
        import traceback
        traceback.print_exc()

def generate_local_lime_explanation(model, x, X_background, feature_names, model_type, 
                                  output_path, position, player_name, predicted_value, max_features=10):
    """Generate local LIME explanation for a single prediction"""
    print(f"Generating local LIME explanation for {player_name} ({position})...")
    
    try:
        # Convert to numpy if needed
        if not isinstance(X_background, np.ndarray):
            X_background = X_background.values
        
        if not isinstance(x, np.ndarray):
            x = x.values
        
        # Flatten if needed
        if hasattr(x, 'flatten'):
            x = x.flatten()
        
        # Create LIME explainer
        lime_explainer = get_lime_explainer(X_background, feature_names)
        if lime_explainer is None:
            return
        
        # Define prediction function based on model type
        if model_type == 'LSTM':
            def predict_fn(x):
                # Reshape for LSTM
                x_3d = np.zeros((1, 3, x.shape[0]))  # Using lookback=3 as default
                for t in range(3):
                    x_3d[0, t, :] = x
                return model.predict(x_3d).flatten()
        else:
            # Standard prediction function
            def predict_fn(x):
                return model.predict(x.reshape(1, -1)).flatten()
        
        # Generate explanation
        explanation = lime_explainer.explain_instance(
            x,
            predict_fn,
            num_features=max_features
        )
        
        # Plot explanation
        plt.figure(figsize=(10, 8))
        explanation.as_pyplot_figure()
        plt.title(f"LIME Explanation - {player_name} ({position})\nPredicted Points: {predicted_value:.2f}")
        plt.tight_layout()
        
        # Clean player name for filename
        clean_name = player_name.replace('/', '_').replace(' ', '_').replace("'", "")
        plt.savefig(os.path.join(output_path, f"lime_explanation_{position}_{clean_name}.png"), dpi=150)
        plt.close()
        
        print(f"  -> Local LIME explanation saved to {output_path}")
    except Exception as e:
        print(f"Error generating local LIME explanation: {e}")

def generate_local_perturbation_explanation(model, x, feature_names, model_type, 
                                           output_path, position, player_name, predicted_value, 
                                           max_features=10, original_data=None):
    """
    Generate local explanation for neural network models using perturbation analysis
    
    Parameters:
    - model: The trained model (LSTM or DeepLearning)
    - x: Feature vector for the specific instance
    - feature_names: List of feature names
    - model_type: Type of model ('LSTM' or 'DeepLearning')
    - output_path: Directory to save visualizations
    - position: Player position (GK, DEF, MID, FWD)
    - player_name: Name of the player
    - predicted_value: The predicted value
    - max_features: Maximum number of features to display
    - original_data: Original unscaled data (for display)
    """
    print(f"Generating local perturbation explanation for {player_name} ({position})...")
    
    try:
        # Ensure x is the right shape
        if x.ndim == 1:
            x_flat = x.copy()
        else:
            x_flat = x.flatten()
        
        # Get original feature values for display
        if original_data is not None:
            if hasattr(original_data, 'values'):
                display_data = original_data.values.flatten()
            else:
                display_data = original_data.flatten() if hasattr(original_data, 'flatten') else original_data
        else:
            display_data = x_flat
        
        # For LSTM models, reshape input appropriately for prediction
        if model_type == 'LSTM':
            lookback = 3  # Default lookback
            x_lstm = np.zeros((1, lookback, len(x_flat)))
            for t in range(lookback):
                x_lstm[0, t, :] = x_flat
            
            # Get original prediction as baseline
            try:
                base_prediction = model.predict(x_lstm)[0][0]
            except Exception as e:
                print(f"Error in base LSTM prediction: {e}")
                base_prediction = predicted_value  # Use provided value as fallback
            
            # Calculate feature contributions using perturbation
            contributions = np.zeros(len(x_flat))
            
            # Perturb each feature
            for i in range(len(x_flat)):
                # Create perturbed feature (zero out to measure contribution)
                x_perturbed = x_flat.copy()
                original_value = x_perturbed[i]
                x_perturbed[i] = 0  # Zero out the feature
                
                # Create LSTM input with perturbed feature
                x_lstm_perturbed = np.zeros((1, lookback, len(x_flat)))
                for t in range(lookback):
                    x_lstm_perturbed[0, t, :] = x_perturbed
                
                # Get prediction with perturbed feature
                try:
                    perturbed_prediction = model.predict(x_lstm_perturbed)[0][0]
                except Exception as e:
                    print(f"Error in perturbed LSTM prediction for feature {i}: {e}")
                    perturbed_prediction = base_prediction  # Fallback to no difference
                
                # Calculate contribution as difference
                contribution = base_prediction - perturbed_prediction
                contributions[i] = contribution
                
                # Restore the original value
                x_perturbed[i] = original_value
        
        else:  # Standard neural network
            # Get original prediction as baseline
            try:
                base_prediction = model.predict(x_flat.reshape(1, -1))[0][0]
            except Exception as e:
                print(f"Error in base prediction: {e}")
                base_prediction = predicted_value  # Use provided value as fallback
            
            # Calculate feature contributions using perturbation
            contributions = np.zeros(len(x_flat))
            
            # Perturb each feature
            for i in range(len(x_flat)):
                # Create perturbed feature (zero out to measure contribution)
                x_perturbed = x_flat.copy()
                original_value = x_perturbed[i]
                x_perturbed[i] = 0  # Zero out the feature
                
                # Get prediction with perturbed feature
                try:
                    perturbed_prediction = model.predict(x_perturbed.reshape(1, -1))[0][0]
                except Exception as e:
                    print(f"Error in perturbed prediction for feature {i}: {e}")
                    perturbed_prediction = base_prediction  # Fallback to no difference
                
                # Calculate contribution as difference
                contribution = base_prediction - perturbed_prediction
                contributions[i] = contribution
                
                # Restore the original value
                x_perturbed[i] = original_value
        
        # Sort features by absolute contribution
        sorted_indices = np.argsort(np.abs(contributions))[::-1]
        top_indices = sorted_indices[:max_features]
        
        # Get top features and their contributions
        top_features = [feature_names[i] for i in top_indices]
        top_contributions = [contributions[i] for i in top_indices]
        top_values = [display_data[i] for i in top_indices]
        
        # Create waterfall-like plot
        plt.figure(figsize=(12, 8))
        
        # Calculate positions for bars
        pos = range(len(top_features))
        
        # Plot bars with different colors for positive and negative contributions
        colors = ['#FF4136' if c > 0 else '#0074D9' for c in top_contributions]
        plt.barh(pos, top_contributions, align='center', color=colors)
        
        # Add feature names with values
        feature_labels = [f"{feature} = {value:.3f}" for feature, value in zip(top_features, top_values)]
        plt.yticks(pos, feature_labels)
        
        # Add base value and final prediction
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Add grid
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add title and labels
        plt.title(f"{model_type} Feature Contributions\n{player_name} ({position})\nPredicted Points: {predicted_value:.2f}")
        plt.xlabel('Feature Contribution')
        
        # Add values to each bar
        for i, v in enumerate(top_contributions):
            plt.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.3f}", 
                     ha='left' if v >= 0 else 'right', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Clean player name for filename
        clean_name = player_name.replace('/', '_').replace(' ', '_').replace("'", "")
        plt.savefig(os.path.join(output_path, f"local_perturbation_{position}_{clean_name}.png"), dpi=150)
        plt.close()
        
        # Create a force plot alternative
        plt.figure(figsize=(12, 4))
        
        # Base value
        base_value = 0  # Start at zero
        cumulative = base_value
        
        # Plot segments
        segments = []
        for i, contrib in enumerate(top_contributions):
            segments.append((cumulative, cumulative + contrib, top_features[i], colors[i]))
            cumulative += contrib
        
        # Sort segments by start position
        segments.sort(key=lambda x: x[0])
        
        # Plot segments
        for start, end, feature, color in segments:
            plt.plot([start, end], [1, 1], color=color, linewidth=20, solid_capstyle='butt')
            
            # Add feature name for significant contributions
            if abs(end - start) > 0.05:
                plt.text((start + end) / 2, 1.1, feature, ha='center', va='bottom', rotation=45)
                plt.text((start + end) / 2, 0.9, f"{end - start:.3f}", ha='center', va='top')
        
        # Add final prediction
        plt.scatter([cumulative], [1], s=100, c='red', zorder=5)
        plt.text(cumulative, 0.8, f"Final: {predicted_value:.2f}", ha='center', va='top', fontweight='bold')
        
        # Remove y-axis and set limits
        plt.yticks([])
        plt.ylim(0.5, 1.5)
        
        # Set x-axis limits with some padding
        x_min = min(min(s[0] for s in segments) - 0.5, -0.5)
        x_max = max(max(s[1] for s in segments) + 0.5, predicted_value + 0.5)
        plt.xlim(x_min, x_max)
        
        plt.title(f"{model_type} Contribution Flow - {player_name} ({position})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"local_force_{position}_{clean_name}.png"), dpi=150)
        plt.close()
        
        print(f"  -> Local perturbation explanation saved to {output_path}")
        
    except Exception as e:
        print(f"Error generating local perturbation explanation: {e}")
        import traceback
        traceback.print_exc()

        
def generate_global_permutation_explanations(model, X, feature_names, model_type, output_path, position, max_features=10):
    """
    Generate global explanations for neural network models using permutation importance
    
    Parameters:
    - model: The trained model (LSTM or DeepLearning)
    - X: Feature data
    - feature_names: List of feature names
    - model_type: Type of model ('LSTM' or 'DeepLearning')
    - output_path: Directory to save visualizations
    - position: Player position (GK, DEF, MID, FWD)
    - max_features: Maximum number of features to display
    """
    print(f"Generating global permutation explanations for {model_type} - {position}...")
    
    try:
        # Sample data if it's too large
        if hasattr(X, 'shape') and X.shape[0] > 200:
            np.random.seed(42)
            indices = np.random.choice(X.shape[0], 200, replace=False)
            X_sample = X[indices] if isinstance(X, np.ndarray) else X.iloc[indices]
        else:
            X_sample = X
        
        # Convert to numpy if needed
        if not isinstance(X_sample, np.ndarray):
            X_sample = X_sample.values
        
        # For LSTM, reshape data appropriately
        if model_type == 'LSTM':
            # Get baseline predictions (handle LSTM 3D input)
            lookback = 3  # Default lookback
            X_lstm = np.zeros((X_sample.shape[0], lookback, X_sample.shape[1]))
            for i in range(X_sample.shape[0]):
                for t in range(lookback):
                    X_lstm[i, t, :] = X_sample[i]
            
            try:
                baseline_preds = model.predict(X_lstm).flatten()
            except Exception as e:
                print(f"Error in baseline LSTM prediction: {e}")
                baseline_preds = np.zeros(X_sample.shape[0])
                for i in range(X_sample.shape[0]):
                    try:
                        baseline_preds[i] = model.predict(X_lstm[i:i+1])[0][0]
                    except:
                        baseline_preds[i] = 0
            
            # Compute feature importance via permutation
            importances = np.zeros(X_sample.shape[1])
            for feat_idx in range(X_sample.shape[1]):
                # Create permuted feature
                X_permuted = X_sample.copy()
                X_permuted[:, feat_idx] = np.random.permutation(X_permuted[:, feat_idx])
                
                # Create LSTM input with permuted feature
                X_lstm_permuted = np.zeros((X_sample.shape[0], lookback, X_sample.shape[1]))
                for i in range(X_sample.shape[0]):
                    for t in range(lookback):
                        X_lstm_permuted[i, t, :] = X_permuted[i]
                
                # Get predictions with permuted feature
                try:
                    permuted_preds = model.predict(X_lstm_permuted).flatten()
                except Exception as e:
                    print(f"Error in permuted prediction for feature {feat_idx}: {e}")
                    permuted_preds = np.zeros(X_sample.shape[0])
                    for i in range(X_sample.shape[0]):
                        try:
                            permuted_preds[i] = model.predict(X_lstm_permuted[i:i+1])[0][0]
                        except:
                            permuted_preds[i] = 0
                
                # Calculate importance as mean absolute difference
                importance = np.mean(np.abs(baseline_preds - permuted_preds))
                importances[feat_idx] = importance
                
                # Status update every 10 features
                if feat_idx % 10 == 0:
                    print(f"  Processed {feat_idx}/{X_sample.shape[1]} features...")
        
        else:  # Standard neural network
            # Get baseline predictions
            try:
                baseline_preds = model.predict(X_sample).flatten()
            except Exception as e:
                print(f"Error in baseline prediction: {e}")
                baseline_preds = np.zeros(X_sample.shape[0])
                for i in range(X_sample.shape[0]):
                    try:
                        baseline_preds[i] = model.predict(X_sample[i:i+1])[0][0]
                    except:
                        baseline_preds[i] = 0
            
            # Compute feature importance via permutation
            importances = np.zeros(X_sample.shape[1])
            for feat_idx in range(X_sample.shape[1]):
                # Create permuted feature
                X_permuted = X_sample.copy()
                X_permuted[:, feat_idx] = np.random.permutation(X_permuted[:, feat_idx])
                
                # Get predictions with permuted feature
                try:
                    permuted_preds = model.predict(X_permuted).flatten()
                except Exception as e:
                    print(f"Error in permuted prediction for feature {feat_idx}: {e}")
                    permuted_preds = np.zeros(X_sample.shape[0])
                    for i in range(X_sample.shape[0]):
                        try:
                            permuted_preds[i] = model.predict(X_permuted[i:i+1])[0][0]
                        except:
                            permuted_preds[i] = 0
                
                # Calculate importance as mean absolute difference
                importance = np.mean(np.abs(baseline_preds - permuted_preds))
                importances[feat_idx] = importance
                
                # Status update every 10 features
                if feat_idx % 10 == 0:
                    print(f"  Processed {feat_idx}/{X_sample.shape[1]} features...")
        
        # Create bar chart of feature importances (sorted)
        indices = np.argsort(importances)[::-1][:max_features]
        top_features = [feature_names[i] for i in indices]
        top_importances = [importances[i] for i in indices]
        
        # Beeswarm-like plot (bar chart)
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_importances, align='center')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Mean Absolute Error Increase')
        plt.title(f"{model_type} Global Feature Importance (Permutation) - {position}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"global_permutation_bar_{position}.png"), dpi=150)
        plt.close()
        
        # Create a second visualization - feature importance heatmap
        plt.figure(figsize=(12, 10))
        plt.barh(range(len(top_features)), top_importances, color='#1E88E5', align='center')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f"{model_type} Feature Importance Ranking - {position}")
        # Add grid lines
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        # Add value labels
        for i, v in enumerate(top_importances):
            plt.text(v + 0.01, i, f"{v:.4f}", va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"global_importance_ranking_{position}.png"), dpi=150)
        plt.close()
        
        # Create correlation heatmap between top features
        if len(top_features) > 1:  # Only if we have multiple important features
            try:
                top_indices = indices[:max_features]
                X_top_features = X_sample[:, top_indices]
                corr_matrix = np.corrcoef(X_top_features.T)
                
                plt.figure(figsize=(10, 8))
                plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                plt.colorbar(label='Correlation')
                plt.xticks(range(len(top_features)), top_features, rotation=90)
                plt.yticks(range(len(top_features)), top_features)
                plt.title(f"{model_type} Top Feature Correlations - {position}")
                
                # Add correlation values to the heatmap
                for i in range(len(top_features)):
                    for j in range(len(top_features)):
                        plt.text(j, i, f"{corr_matrix[i, j]:.2f}", 
                                ha="center", va="center", 
                                color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, f"global_feature_correlations_{position}.png"), dpi=150)
                plt.close()
            except Exception as e:
                print(f"Error creating correlation heatmap: {e}")
        
        print(f"  -> Global permutation explanations saved to {output_path}")
        return importances, feature_names
        
    except Exception as e:
        print(f"Error generating global permutation explanations: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_model_explanations(
    regression_models, 
    df_pred, 
    drop_cols, 
    output_dir, 
    explain_global=True, 
    explain_local=True, 
    explain_method='shap', 
    max_features=10
):
    """Generate all explanations for regression models, using custom methods for neural networks"""
    # Get unique algorithms
    algos = set(algo for (pos_label, algo) in regression_models.keys())
    
    for algo_name in algos:
        print(f"\nProcessing explanations for {algo_name} models...")
        
        # Create output directories
        _, local_dir, global_dir = create_explanation_directories(output_dir, algo_name)
        
        # Process each position
        for pos_label in ['FWD', 'MID', 'DEF', 'GK']:
            key = (pos_label, algo_name)
            if key not in regression_models:
                continue
            
            model_dict = regression_models[key]
            model = model_dict["model"]
            scaler = model_dict["scaler"]
            
            # Filter data for this position
            df_pos = df_pred[df_pred["position_label"] == pos_label].copy()
            if df_pos.empty:
                continue
            
            # Prepare feature data
            X_reg = df_pos.drop(columns=drop_cols + ["position_label"], errors='ignore')
            X_reg_scaled = scaler.transform(X_reg)
            feature_names = list(X_reg.columns)
            
            # Generate global explanations if requested
            if explain_global:
                # For neural network models, use custom permutation importance
                if algo_name in ['DeepLearning', 'LSTM'] or 'keras' in str(type(model)).lower():
                    importances, _ = generate_global_permutation_explanations(
                        model, X_reg_scaled, feature_names, algo_name, 
                        global_dir, pos_label, max_features
                    )
                else:
                    # For non-neural network models, use SHAP as before
                    if explain_method in ['shap', 'both']:
                        generate_global_shap_explanations(
                            model, X_reg_scaled, feature_names, algo_name, 
                            global_dir, pos_label, max_features
                        )
                    
                    if explain_method in ['lime', 'both']:
                        generate_global_lime_explanations(
                            model, X_reg_scaled, feature_names, algo_name,
                            global_dir, pos_label, max_features
                        )
                
                # Export feature importance to CSV
                export_feature_importance_to_csv(
                    model, X_reg_scaled, feature_names, algo_name,
                    global_dir, pos_label
                )
            
            # Make predictions for local explanations
            if explain_local:
                # Check if it's an LSTM model
                is_lstm = algo_name == 'LSTM'
                lookback = model_dict.get('lookback', 3) if is_lstm else None
                
                if is_lstm:
                    # For LSTM, reshape data appropriately
                    X_lstm = np.zeros((X_reg_scaled.shape[0], lookback, X_reg_scaled.shape[1]))
                    for i in range(X_reg_scaled.shape[0]):
                        for t in range(lookback):
                            X_lstm[i, t, :] = X_reg_scaled[i]
                    
                    try:
                        y_pred = model.predict(X_lstm).flatten()
                    except:
                        # Try prediction one sample at a time
                        y_pred = np.zeros(X_reg_scaled.shape[0])
                        for i in range(X_reg_scaled.shape[0]):
                            try:
                                y_pred[i] = model.predict(X_lstm[i:i+1])[0][0]
                            except Exception as e:
                                print(f"Error in LSTM prediction: {e}")
                                y_pred[i] = 3.0  # Default
                else:
                    # Standard prediction
                    y_pred = model.predict(X_reg_scaled).flatten()
                
                # Add predictions to dataframe
                df_pos['predicted_points'] = y_pred
                
                # Find top player
                top_idx = df_pos['predicted_points'].idxmax()
                top_player = df_pos.loc[top_idx]
                x_top = X_reg.loc[top_idx]  # Original unscaled data
                x_top_scaled = X_reg_scaled[df_pos.index.get_loc(top_idx)]  # Scaled data
                
                print(f"Top {pos_label} player: {top_player['name']} with predicted points: {top_player['predicted_points']:.2f}")
                
                # Generate local explanations
                # For neural network models, use custom perturbation approach
                if algo_name in ['DeepLearning', 'LSTM'] or 'keras' in str(type(model)).lower():
                    generate_local_perturbation_explanation(
                        model, x_top_scaled, feature_names, algo_name,
                        local_dir, pos_label, top_player['name'], 
                        top_player['predicted_points'], max_features,
                        original_data=x_top
                    )
                else:
                    # For non-neural network models, use SHAP as before
                    if explain_method in ['shap', 'both']:
                        generate_local_shap_explanation(
                            model, x_top_scaled, X_reg_scaled, feature_names, algo_name,
                            local_dir, pos_label, top_player['name'], 
                            top_player['predicted_points'], max_features,
                            original_data=x_top
                        )
                    
                    if explain_method in ['lime', 'both']:
                        generate_local_lime_explanation(
                            model, x_top_scaled, X_reg_scaled, feature_names, algo_name,
                            local_dir, pos_label, top_player['name'],
                            top_player['predicted_points'], max_features
                        )

def export_feature_importance_to_csv(model, X, feature_names, model_type, output_dir, pos_label):
    """
    Extract and export global feature importance values to CSV
    
    Parameters:
    - model: The trained model
    - X: Feature data
    - feature_names: List of feature names
    - model_type: Type of model (e.g., 'RFReg', 'GBReg', 'LSTM')
    - output_dir: Directory to save the CSV
    - pos_label: Position label (GK, DEF, MID, FWD)
    """
    print(f"Exporting feature importance values for {model_type} - {pos_label}...")
    
    try:
        # Create a DataFrame to store the values
        importance_df = pd.DataFrame({'feature': feature_names})
        
        # Get feature importance based on model type
        if model_type in ['RFReg', 'GBReg', 'XGBoost'] or hasattr(model, 'feature_importances_'):
            # Tree-based models have feature_importances_ attribute
            importances = model.feature_importances_
            importance_df['importance'] = importances
            importance_df['importance_type'] = 'native'
            
        elif model_type in ['Ridge', 'LinearRegression'] or 'linear' in str(type(model)).lower():
            # Linear models have coefficients
            coeffs = model.coef_
            if len(coeffs) != len(feature_names):
                # Handle multi-output or reshaped coefficients
                coeffs = coeffs.flatten()
                # Truncate or pad if needed
                if len(coeffs) > len(feature_names):
                    coeffs = coeffs[:len(feature_names)]
                elif len(coeffs) < len(feature_names):
                    coeffs = np.pad(coeffs, (0, len(feature_names) - len(coeffs)))
            
            importance_df['importance'] = np.abs(coeffs)  # Use absolute values for importance
            importance_df['importance_type'] = 'coefficient'
            
        elif model_type in ['DeepLearning', 'LSTM'] or 'keras' in str(type(model)).lower():
            # For neural networks, use permutation importance - this is model-agnostic and works
            # for any model regardless of its internal structure
            print(f"Computing permutation importance for {model_type} model...")
            
            # Sample data for efficiency
            if hasattr(X, 'shape') and X.shape[0] > 200:
                np.random.seed(42)
                indices = np.random.choice(X.shape[0], 200, replace=False)
                X_sample = X[indices] if isinstance(X, np.ndarray) else X.iloc[indices]
            else:
                X_sample = X
            
            # Convert to numpy if needed
            if not isinstance(X_sample, np.ndarray):
                X_sample = X_sample.values
            
            # If it's an LSTM model, reshape the input data appropriately
            if model_type == 'LSTM':
                # Setup prediction function for LSTM
                lookback = 3  # Default lookback
                
                # Create 3D tensor for LSTM input (samples, timesteps, features)
                X_lstm = np.zeros((X_sample.shape[0], lookback, X_sample.shape[1]))
                for i in range(X_sample.shape[0]):
                    for t in range(lookback):
                        X_lstm[i, t, :] = X_sample[i]
                
                # Get baseline predictions
                try:
                    baseline_preds = model.predict(X_lstm).flatten()
                except Exception as e:
                    print(f"Error in baseline LSTM prediction: {e}")
                    # Try one at a time
                    baseline_preds = np.zeros(X_sample.shape[0])
                    for i in range(X_sample.shape[0]):
                        try:
                            pred = model.predict(X_lstm[i:i+1])
                            if hasattr(pred, 'flatten'):
                                baseline_preds[i] = pred.flatten()[0]
                            else:
                                baseline_preds[i] = float(pred[0][0])
                        except Exception as e2:
                            print(f"Error in individual prediction {i}: {e2}")
                            baseline_preds[i] = 0.0
                
                # Compute feature importance via permutation for each feature
                importances = np.zeros(X_sample.shape[1])
                for feat_idx in range(X_sample.shape[1]):
                    # Create permuted feature
                    X_permuted = X_sample.copy()
                    
                    # Permute this feature
                    X_permuted[:, feat_idx] = np.random.permutation(X_permuted[:, feat_idx])
                    
                    # Create LSTM input with permuted feature
                    X_lstm_permuted = np.zeros((X_sample.shape[0], lookback, X_sample.shape[1]))
                    for i in range(X_sample.shape[0]):
                        for t in range(lookback):
                            X_lstm_permuted[i, t, :] = X_permuted[i]
                    
                    # Get predictions with permuted feature
                    try:
                        permuted_preds = model.predict(X_lstm_permuted).flatten()
                    except Exception as e:
                        print(f"Error in permuted prediction for feature {feat_idx}: {e}")
                        # Try one at a time
                        permuted_preds = np.zeros(X_sample.shape[0])
                        for i in range(X_sample.shape[0]):
                            try:
                                pred = model.predict(X_lstm_permuted[i:i+1])
                                if hasattr(pred, 'flatten'):
                                    permuted_preds[i] = pred.flatten()[0]
                                else:
                                    permuted_preds[i] = float(pred[0][0])
                            except:
                                permuted_preds[i] = 0.0
                    
                    # Calculate importance as mean absolute difference
                    importance = np.mean(np.abs(baseline_preds - permuted_preds))
                    importances[feat_idx] = importance
                    
                    # Status update every 10 features
                    if feat_idx % 10 == 0:
                        print(f"  Processed {feat_idx}/{X_sample.shape[1]} features...")
                
                print(f"Completed permutation importance for all {X_sample.shape[1]} features")
                
                importance_df['importance'] = importances
                importance_df['importance_type'] = 'permutation'
                
            else:  # Standard neural network (not LSTM)
                # Get baseline predictions
                try:
                    baseline_preds = model.predict(X_sample).flatten()
                except Exception as e:
                    print(f"Error in baseline prediction: {e}")
                    # Try one at a time
                    baseline_preds = np.zeros(X_sample.shape[0])
                    for i in range(X_sample.shape[0]):
                        try:
                            pred = model.predict(X_sample[i:i+1])
                            if hasattr(pred, 'flatten'):
                                baseline_preds[i] = pred.flatten()[0]
                            else:
                                baseline_preds[i] = float(pred[0])
                        except:
                            baseline_preds[i] = 0.0
                
                # Compute feature importance via permutation for each feature
                importances = np.zeros(X_sample.shape[1])
                for feat_idx in range(X_sample.shape[1]):
                    # Create permuted feature
                    X_permuted = X_sample.copy()
                    
                    # Permute this feature
                    X_permuted[:, feat_idx] = np.random.permutation(X_permuted[:, feat_idx])
                    
                    # Get predictions with permuted feature
                    try:
                        permuted_preds = model.predict(X_permuted).flatten()
                    except Exception as e:
                        print(f"Error in permuted prediction for feature {feat_idx}: {e}")
                        # Try one at a time
                        permuted_preds = np.zeros(X_sample.shape[0])
                        for i in range(X_sample.shape[0]):
                            try:
                                pred = model.predict(X_permuted[i:i+1])
                                if hasattr(pred, 'flatten'):
                                    permuted_preds[i] = pred.flatten()[0]
                                else:
                                    permuted_preds[i] = float(pred[0])
                            except:
                                permuted_preds[i] = 0.0
                    
                    # Calculate importance as mean absolute difference
                    importance = np.mean(np.abs(baseline_preds - permuted_preds))
                    importances[feat_idx] = importance
                    
                    # Status update every 10 features
                    if feat_idx % 10 == 0:
                        print(f"  Processed {feat_idx}/{X_sample.shape[1]} features...")
                
                print(f"Completed permutation importance for all {X_sample.shape[1]} features")
                
                importance_df['importance'] = importances
                importance_df['importance_type'] = 'permutation'
                
        else:
            # Fallback for other model types - use permutation importance
            from sklearn.inspection import permutation_importance
            
            # Sample data for efficiency
            if hasattr(X, 'shape') and X.shape[0] > 200:
                np.random.seed(42)
                indices = np.random.choice(X.shape[0], 200, replace=False)
                X_sample = X[indices] if isinstance(X, np.ndarray) else X.iloc[indices]
            else:
                X_sample = X
            
            # Convert to numpy if needed
            if not isinstance(X_sample, np.ndarray):
                X_sample = X_sample.values
            
            # Try standard prediction function
            try:
                perm_result = permutation_importance(
                    model, X_sample, model.predict(X_sample),
                    n_repeats=5, random_state=42
                )
                importances = perm_result.importances_mean
                importance_df['importance'] = importances
                importance_df['importance_type'] = 'permutation'
            except Exception as e:
                print(f"Error calculating permutation importance: {e}")
                # Fall back to custom permutation implementation
                baseline_preds = model.predict(X_sample)
                
                importances = np.zeros(X_sample.shape[1])
                for feat_idx in range(X_sample.shape[1]):
                    X_permuted = X_sample.copy()
                    X_permuted[:, feat_idx] = np.random.permutation(X_permuted[:, feat_idx])
                    permuted_preds = model.predict(X_permuted)
                    importance = np.mean(np.abs(baseline_preds - permuted_preds))
                    importances[feat_idx] = importance
                
                importance_df['importance'] = importances
                importance_df['importance_type'] = 'manual_permutation'
        
        # Sort by importance (descending)
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Add metadata columns
        importance_df['model'] = model_type
        importance_df['position'] = pos_label
        
        # If the GW is available in the output path, extract it
        try:
            import re
            gw_match = re.search(r'gw(\d+)', output_dir)
            if gw_match:
                gw = gw_match.group(1)
                importance_df['gameweek'] = gw
        except:
            pass
        
        # Save to CSV
        csv_path = os.path.join(output_dir, f"feature_importance_{model_type}_{pos_label}.csv")
        importance_df.to_csv(csv_path, index=False)
        print(f"  -> Saved feature importance CSV to {csv_path}")
        
        return importance_df
    
    except Exception as e:
        print(f"Error exporting feature importance: {e}")
        import traceback
        traceback.print_exc()
        return None

# ------------------- Original Helper Functions -------------------

def get_random_subset(X, max_rows=200, random_state=42):
    """Return up to max_rows from X (which can be a numpy array or dataframe)."""
    if X.shape[0] <= max_rows:
        return X
    random.seed(random_state)
    indices = random.sample(range(X.shape[0]), max_rows)
    if isinstance(X, pd.DataFrame):
        return X.iloc[indices]
    else:
        return X[indices]

def ensure_scalar(value, default=0.0):
    """
    Ensures the input value is a scalar float, regardless of input type.
    
    Parameters:
    - value: The value to convert (could be scalar, array, list, etc.)
    - default: Default value to use if conversion fails
    
    Returns:
    - A scalar float
    """
    try:
        if isinstance(value, (list, tuple)):
            return float(value[0])
        elif isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            return float(value.flatten()[0])
        else:
            return float(value)
    except (TypeError, IndexError, ValueError):
        return default

def extract_single_prediction(prediction_result):
    """
    Safely extracts a single scalar prediction from various model outputs.
    
    Parameters:
    - prediction_result: Prediction result from model.predict()
    
    Returns:
    - A scalar float representing the prediction
    """
    try:
        if isinstance(prediction_result, (list, tuple)):
            return float(prediction_result[0])
        elif isinstance(prediction_result, np.ndarray):
            if prediction_result.ndim > 1:
                # For multi-dimensional arrays, get the first element
                return float(prediction_result.flatten()[0])
            else:
                # For 1D arrays
                return float(prediction_result[0])
        else:
            return float(prediction_result)
    except (IndexError, ValueError, TypeError):
        # Fallback to a reasonable default
        return 3.0  # Average FPL points

def prediction_safely(model, data, reshape_for_lstm=False, lookback=3):
    """
    Makes predictions safely, handling various model types and input shapes.
    
    Parameters:
    - model: ML model
    - data: Input data
    - reshape_for_lstm: Whether to reshape for LSTM models
    - lookback: Number of lookback steps for LSTM
    
    Returns:
    - A scalar prediction value
    """
    try:
        if reshape_for_lstm:
            # Create LSTM input with proper lookback
            if data.ndim == 2:
                # Assume (samples, features)
                samples, features = data.shape
                lstm_input = np.zeros((samples, lookback, features), dtype=np.float32)
                # Fill with the same data for each timestep
                for t in range(lookback):
                    lstm_input[:, t, :] = data
            else:
                # Single sample
                features = data.shape[0]
                lstm_input = np.zeros((1, lookback, features), dtype=np.float32)
                for t in range(lookback):
                    lstm_input[0, t, :] = data
            
            # Make prediction
            pred = model.predict(lstm_input)
        else:
            # Standard prediction
            pred = model.predict(data)
        
        # Extract scalar value
        return extract_single_prediction(pred)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 3.0  # Fallback to average FPL points

def extract_base_value(base_val, class_index=1):
    """
    For classification with multiple classes, explainer.expected_value can be:
      - A list of length n_classes,
      - A numpy array shape (n_classes,),
      - A single scalar.
    We want a single float for class_index=1. 
    """
    if isinstance(base_val, list):
        if len(base_val) > class_index:
            return float(base_val[class_index])
        else:
            return float(base_val[0])
    elif isinstance(base_val, np.ndarray):
        if base_val.ndim == 1 and base_val.shape[0] > class_index:
            return float(base_val[class_index])
        else:
            return float(base_val[0])
    else:
        return float(base_val)

def prepare_lstm_sequences(X_data, pos_label, model_dict):
    """
    Prepare data for LSTM prediction using the sequence information stored in the model dict.
    
    Parameters:
    - X_data: DataFrame with features
    - pos_label: Position label (for metadata lookup)
    - model_dict: Dictionary containing model and sequence information
    
    Returns:
    - X_lstm: Numpy array with shape (samples, lookback, features)
    """
    # Extract sequence information from model dict
    lookback = model_dict.get("lookback", 3)  # Default to lookback of 3
    feature_mapping = model_dict.get("feature_mapping", {})
    
    # If no feature mapping found, use a simplified approach
    if not feature_mapping:
        print(f"Warning: No feature mapping found for LSTM model for {pos_label}")
        # Ensure data is numeric and reshape to default format
        X_numeric = X_data.select_dtypes(include=[np.number]).copy()
        return X_numeric.values.reshape((X_numeric.shape[0], 1, X_numeric.shape[1]))
    
    # Convert feature mapping keys to integers if they're strings
    feature_mapping = {int(k) if isinstance(k, str) else k: v for k, v in feature_mapping.items()}
    
    # Get number of features in the LSTM model
    num_features = len(feature_mapping)
    
    # Create a 3D array for LSTM input
    num_samples = X_data.shape[0]
    X_lstm = np.zeros((num_samples, lookback, num_features), dtype=np.float32)
    
    # Fill the array with available features
    for i in range(num_samples):
        # For each time step
        for t in range(lookback):
            # For each feature index in mapping
            for feat_idx, feat_name in feature_mapping.items():
                if feat_name in X_data.columns:
                    # Use the same value for all time steps (limitation for prediction)
                    X_lstm[i, t, feat_idx] = X_data.iloc[i][feat_name]
    
    return X_lstm

# Helper function to create the output directory for a specific gameweek
def create_gameweek_output_dir(base_output_dir, gw):
    """Create the output directory for a specific gameweek"""
    gw_dir = os.path.join(base_output_dir, f"gw{gw:02d}")
    os.makedirs(gw_dir, exist_ok=True)
    return gw_dir

# Helper function to get the dataset filename for a specific gameweek
def get_gameweek_dataset_path(data_dir, gw):
    """Get the path to the dataset file for a specific gameweek"""
    return os.path.join(data_dir, f"player_predictions_data_gw{gw:02d}.csv")

def process_gameweek_wrapper(*args, **kwargs):
    """Wrapper function to catch any exceptions in child processes"""
    try:
        return process_gameweek(*args, **kwargs)
    except Exception as e:
        import traceback
        error_message = f"Error processing gameweek {args[0]}: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_message, "gameweek": args[0]}

# Main function to process a single gameweek
def process_gameweek(
    gw, 
    data_dir, 
    output_base_dir, 
    regression_models, 
    classification_models, 
    metadata=None,
    include_all_players=False,
    skip_explain=False,
    explain_global=True,
    explain_local=True,
    explain_method='shap',
    max_features=10
):
    """Process predictions for a single gameweek"""
    import matplotlib
    matplotlib.use('Agg')  # Use the 'Agg' backend which doesn't require GUI
    import matplotlib.pyplot as plt

    print(f"\n{'='*40}")
    print(f"Processing Gameweek {gw}")
    print(f"{'='*40}")
    
    start_time = time.time()
    
    # Create gameweek-specific output directory
    output_dir = create_gameweek_output_dir(output_base_dir, gw)
    
    # Get dataset path
    dataset_path = get_gameweek_dataset_path(data_dir, gw)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"WARNING: Dataset for gameweek {gw} not found at {dataset_path}")
        return None
    
    # Load prediction data
    df_pred = pd.read_csv(dataset_path)
    df_pred['position_label'] = df_pred.apply(get_position_label, axis=1)
    df_pred = df_pred[df_pred['position_label'] != 'UNK']
    
    if df_pred.empty:
        print(f"No valid rows found for gameweek {gw}. Skipping.")
        return None
    
    # Check for fixture information
    if 'has_fixture_next_week' in df_pred.columns:
        has_fixture_count = df_pred['has_fixture_next_week'].sum()
        no_fixture_count = len(df_pred) - has_fixture_count
        print(f"\nPlayers with fixtures next week: {has_fixture_count}")
        print(f"Players without fixtures next week: {no_fixture_count}")
        
        # Filter out players without fixtures unless include_all_players is specified
        if not include_all_players:
            original_count = len(df_pred)
            df_pred = df_pred[df_pred['has_fixture_next_week'] == 1]
            filtered_count = len(df_pred)
            print(f"Filtered out {original_count - filtered_count} players without fixtures")
            
            if df_pred.empty:
                print(f"ERROR: No players with fixtures found for gameweek {gw}. Skipping.")
                return None
    else:
        print("WARNING: 'has_fixture_next_week' column not found in dataset. All players will be included.")
    
    # Columns to drop for regression
    drop_for_reg = [
        'season_x','team_x','name','kickoff_time','opp_team_name','game_date',
        'next_match_points','played_next_match','started_next_match','next_fixture',
        'has_fixture_next_week'  # Added this column to be dropped before prediction
    ]
    
    # Process regression models
    print(f"Processing regression models for gameweek {gw}...")
    
    # Dictionary to store results for this gameweek
    all_predictions = {}
    
    # Get all unique regression algorithms
    reg_algos = set(algo for (pos_label, algo) in regression_models.keys())
    
    for algo_name in reg_algos:
        # Skip SVM as it's no longer used
        if algo_name == "SVM":
            continue
        
        print(f"\nProcessing {algo_name} models...")
        
        # Create folder structure
        algo_output_dir = os.path.join(output_dir, algo_name)
        pred_dir = os.path.join(algo_output_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        
        all_preds = []
        
        for pos_label in ['FWD','MID','DEF','GK']:
            key = (pos_label, algo_name)
            if key not in regression_models:
                continue
            
            model_dict = regression_models[key]
            model = model_dict["model"]
            scaler = model_dict["scaler"]
            
            df_pos = df_pred[df_pred["position_label"] == pos_label].copy()
            if df_pos.empty:
                continue
            
            # Prepare X - make sure to drop has_fixture_next_week
            X_reg = df_pos.drop(columns=drop_for_reg + ["position_label"], errors='ignore')
            X_reg_scaled = scaler.transform(X_reg)
            
            # Determine if LSTM reshape is needed
            is_lstm = algo_name == "LSTM"
            lookback = model_dict.get("lookback", 3) if is_lstm else 3
            
            # Make predictions for each player
            y_pred = []
            for i in range(X_reg_scaled.shape[0]):
                x_i = X_reg_scaled[i:i+1]  # Get single sample
                pred = prediction_safely(model, x_i, is_lstm, lookback)
                y_pred.append(pred)
            
            # Add predictions to dataframe
            df_pos["predicted_points"] = y_pred
            
            # Add gameweek information
            df_pos["gameweek"] = gw
            
            # Add fixture status for display purposes
            if 'has_fixture_next_week' in df_pos.columns:
                # Store has_fixture_next_week status for output
                df_pos["has_fixture"] = df_pos['has_fixture_next_week'].map({1: 'Yes', 0: 'No'})
            
            all_preds.append(df_pos)
        
        if not all_preds:
            print(f"No predictions found for {algo_name} in gameweek {gw}, skipping.")
            continue
        
        # Combine predictions across positions
        combined = pd.concat(all_preds, ignore_index=True)
        
        # Sort predictions by position and score
        def pos_sort_key(pos):
            return {'FWD': 0, 'MID': 1, 'DEF': 2, 'GK': 3}.get(pos, 4)
            
        combined.sort_values(
            by=["position_label","predicted_points"],
            key=lambda col: col.map(pos_sort_key) if col.name=="position_label" else col,
            ascending=[True, False],
            inplace=True
        )
        
        # Save predictions - now including gameweek and fixture status if available
        cols_to_save = ["name", "position_label", "predicted_points", "gameweek"]
        if 'has_fixture' in combined.columns:
            cols_to_save.append("has_fixture")
        
        out_csv = os.path.join(pred_dir, f"predictions_{algo_name}_regression.csv")
        combined[cols_to_save].to_csv(out_csv, index=False)
        print(f"Saved predictions => {out_csv}")
        
        # Store for later use
        all_predictions[algo_name] = combined
    
    # Generate explanations if not skipped
    if not skip_explain:
        # Determine which types of explanations to generate
        if explain_global or explain_local:
            do_explain_global = explain_global
            do_explain_local = explain_local
        else:
            # If neither specified, generate both
            do_explain_global = True
            do_explain_local = True
            
        generate_model_explanations(
            regression_models,
            df_pred,
            drop_for_reg,
            output_dir,
            explain_global=do_explain_global,
            explain_local=do_explain_local,
            explain_method=explain_method,
            max_features=max_features
        )
    else:
        print(f"Skipping explainability figure generation for gameweek {gw} as requested.")
    
    # Process classification models
    print(f"\nProcessing classification models for gameweek {gw}...")

    # For classification models, also ensure has_fixture_next_week is not used
    drop_for_class = [
        'season_x','team_x','name','kickoff_time','opp_team_name','game_date',
        'played_next_match','started_next_match','next_fixture','next_match_points',
        'has_fixture_next_week'  # Added this column
    ]

    classification_tasks = ["played_next", "started_next"]
    
    # Dictionary to store classification results
    class_predictions = {}

    for task in classification_tasks:
        relevant_keys = [k for k in classification_models.keys() if k[1] == task]
        if not relevant_keys:
            print(f"No classification models found for task={task}, skipping.")
            continue
        
        clf_algos = set(k[2] for k in relevant_keys)
        for clf_algo_name in clf_algos:
            
            algo_output_dir = os.path.join(output_dir, f"{clf_algo_name}_{task}")
            pred_dir = os.path.join(algo_output_dir, "predictions")
            os.makedirs(pred_dir, exist_ok=True)
            
            # Combine classification preds
            preds_list = []
            for key in relevant_keys:
                pos_label, _, clf_algo = key
                if clf_algo != clf_algo_name:
                    continue
                
                model_dict = classification_models[key]
                model = model_dict["model"]
                scaler = model_dict["scaler"]
                
                df_pos = df_pred[df_pred['position_label'] == pos_label].copy()
                if df_pos.empty:
                    continue
                
                # Make sure to drop has_fixture_next_week
                X_class = df_pos.drop(columns=drop_for_class + ["position_label"], errors='ignore')
                X_class_scaled = scaler.transform(X_class)
                
                # Get model predictions and probability scores
                try:
                    prob_1 = model.predict_proba(X_class_scaled)[:,1]
                    label_pred = (prob_1 >= 0.5).astype(int)
                except Exception as e:
                    print(f"Error getting predictions for {clf_algo_name}, {task}, {pos_label}: {e}")
                    continue
                
                df_pos[f"{task}_prob"] = prob_1
                df_pos[f"{task}_pred"] = label_pred
                df_pos["clf_algo"] = clf_algo_name
                
                # Add gameweek information
                df_pos["gameweek"] = gw
                
                # Add fixture status for display purposes
                if 'has_fixture_next_week' in df_pos.columns:
                    df_pos["has_fixture"] = df_pos['has_fixture_next_week'].map({1: 'Yes', 0: 'No'})
                
                preds_list.append(df_pos)
            
            if not preds_list:
                continue
            
            combined_class = pd.concat(preds_list, ignore_index=True)
            def pos_order(pos):
                return POSITION_ORDER[pos]
            
            combined_class.sort_values(
                by=["position_label", f"{task}_prob"],
                key=lambda col: col.map(pos_order) if col.name=="position_label" else col,
                ascending=[True, False],
                inplace=True
            )
            
            # Save predictions with gameweek and fixture status if available
            cols_to_save = ["name", "position_label", f"{task}_prob", f"{task}_pred", "gameweek"]
            if 'has_fixture' in combined_class.columns:
                cols_to_save.append("has_fixture")
                
            out_csv_class = os.path.join(pred_dir, f"predictions_{clf_algo_name}_{task}.csv")
            combined_class[cols_to_save].to_csv(out_csv_class, index=False)
            print(f"Saved classification predictions => {out_csv_class}")
            
            # Store for later use
            class_predictions[f"{clf_algo_name}_{task}"] = combined_class

    elapsed_time = time.time() - start_time
    print(f"\nCompleted gameweek {gw} processing in {elapsed_time:.1f} seconds")
    
    # Return both regression and classification predictions for this gameweek
    return {
        "gameweek": gw,
        "regression": all_predictions,
        "classification": class_predictions
    }

# Function to create summary files combining all processed gameweeks
def create_summary_files(gameweek_results, output_dir):
    """
    Create summary files combining results from all processed gameweeks
    
    Parameters:
    - gameweek_results: List of dictionaries containing results for each gameweek
    - output_dir: Base output directory
    """
    print("\nCreating summary files for all processed gameweeks...")
    
    if not gameweek_results:
        print("No gameweek results to summarize.")
        return
    
    # Create a summary directory
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Process regression model results
    for algo_name in gameweek_results[0]["regression"].keys():
        # Combine results from all gameweeks
        all_gw_preds = []
        for gw_result in gameweek_results:
            if algo_name in gw_result["regression"]:
                all_gw_preds.append(gw_result["regression"][algo_name])
        
        if not all_gw_preds:
            continue
        
        # Combine into a single DataFrame
        combined = pd.concat(all_gw_preds, ignore_index=True)
        
        # Save summary file
        summary_file = os.path.join(summary_dir, f"summary_{algo_name}_regression.csv")
        combined.to_csv(summary_file, index=False)
        print(f"Saved regression summary => {summary_file}")
        
        # Create a pivot table showing predictions by player across gameweeks
        try:
            pivot_df = combined.pivot_table(
                index=["name", "position_label"],
                columns="gameweek",
                values="predicted_points",
                aggfunc='first'
            ).reset_index()
            
            # Calculate player averages across gameweeks
            pivot_df["avg_points"] = pivot_df.iloc[:, 2:].mean(axis=1)
            
            # Sort by average points
            pivot_df = pivot_df.sort_values(by=["position_label", "avg_points"], ascending=[True, False])
            
            # Save pivot table
            pivot_file = os.path.join(summary_dir, f"summary_pivot_{algo_name}_regression.csv")
            pivot_df.to_csv(pivot_file, index=False)
            print(f"Saved regression pivot table => {pivot_file}")
        except Exception as e:
            print(f"Error creating pivot table for {algo_name}: {e}")
    
    # Process classification model results
    class_keys = set()
    for gw_result in gameweek_results:
        class_keys.update(gw_result["classification"].keys())
    
    for key in class_keys:
        # Combine results from all gameweeks
        all_gw_preds = []
        for gw_result in gameweek_results:
            if key in gw_result["classification"]:
                all_gw_preds.append(gw_result["classification"][key])
        
        if not all_gw_preds:
            continue
        
        # Combine into a single DataFrame
        combined = pd.concat(all_gw_preds, ignore_index=True)
        
        # Extract task and algorithm from key
        parts = key.split("_")
        task = parts[-1]
        algo_name = "_".join(parts[:-1])
        
        # Save summary file
        summary_file = os.path.join(summary_dir, f"summary_{algo_name}_{task}.csv")
        combined.to_csv(summary_file, index=False)
        print(f"Saved classification summary => {summary_file}")
        
        # Create a pivot table showing predictions by player across gameweeks
        try:
            pivot_df = combined.pivot_table(
                index=["name", "position_label"],
                columns="gameweek",
                values=f"{task}_prob",
                aggfunc='first'
            ).reset_index()
            
            # Calculate player averages across gameweeks
            pivot_df["avg_prob"] = pivot_df.iloc[:, 2:].mean(axis=1)
            
            # Sort by average probability
            pivot_df = pivot_df.sort_values(by=["position_label", "avg_prob"], ascending=[True, False])
            
            # Save pivot table
            pivot_file = os.path.join(summary_dir, f"summary_pivot_{algo_name}_{task}.csv")
            pivot_df.to_csv(pivot_file, index=False)
            print(f"Saved classification pivot table => {pivot_file}")
        except Exception as e:
            print(f"Error creating pivot table for {key}: {e}")

# Function to find available gameweek datasets
def find_available_gameweek_datasets(data_dir):
    """Find all available gameweek datasets and return their gameweek numbers"""
    pattern = os.path.join(data_dir, "player_predictions_data_gw*.csv")
    files = glob.glob(pattern)
    
    gameweeks = []
    for file in files:
        match = re.search(r'gw(\d+)\.csv$', file)
        if match:
            gw = int(match.group(1))
            gameweeks.append(gw)
    
    return sorted(gameweeks)

# ------------------- Main Script -------------------

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    script_dir = os.path.dirname(__file__)
    model_dir = os.path.join(script_dir, "..", "models")
    data_dir = os.path.join(script_dir, "..", "data")
    
    # Set base output directory
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        base_output_dir = os.path.join(script_dir, "..", "predictions")
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    with open(os.path.join(model_dir, "regression_models.pkl"), "rb") as f:
        regression_models = pickle.load(f)
    
    with open(os.path.join(model_dir, "classification_models.pkl"), "rb") as f:
        classification_models = pickle.load(f)
    
    # Try to load metadata
    metadata = {}
    metadata_path = os.path.join(model_dir, "metadata", "feature_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print("Loaded model metadata.")
    
    # Determine which gameweeks to process
    gameweeks_to_process = []
    
    if args.gw:
        # Process a single gameweek
        gameweeks_to_process = [args.gw]
    elif args.gw_range:
        # Process a range of gameweeks
        try:
            start_gw, end_gw = map(int, args.gw_range.split('-'))
            gameweeks_to_process = list(range(start_gw, end_gw + 1))
        except:
            print(f"Error parsing gameweek range: {args.gw_range}")
            print("Format should be 'start-end', e.g., '1-10'")
            return
    elif args.all_gws:
        # Process all available gameweeks
        available_gws = find_available_gameweek_datasets(data_dir)
        if not available_gws:
            print("No gameweek datasets found. Please run gather_prediction_data.py first.")
            return
        gameweeks_to_process = available_gws
    else:
        # Default to processing the latest available gameweek
        available_gws = find_available_gameweek_datasets(data_dir)
        if not available_gws:
            print("No gameweek datasets found. Please run gather_prediction_data.py first.")
            return
        gameweeks_to_process = [max(available_gws)]
    
    print(f"Will process gameweeks: {gameweeks_to_process}")
    
    # Process each gameweek
    all_results = []
    
    if args.parallel and len(gameweeks_to_process) > 1:
        print(f"Processing {len(gameweeks_to_process)} gameweeks in parallel...")
        
        # Determine number of processes
        max_processes = args.max_processes if args.max_processes else min(len(gameweeks_to_process), multiprocessing.cpu_count())
        print(f"Using up to {max_processes} processes")
        
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            # Submit all gameweek processing tasks
            futures = {
                executor.submit(
                    process_gameweek_wrapper,
                    gw,
                    data_dir,
                    base_output_dir,
                    regression_models,
                    classification_models,
                    metadata,
                    args.include_all_players,
                    args.skip_explain,
                    args.explain_global,
                    args.explain_local,
                    args.explain_method,
                    args.max_features
                ): gw for gw in gameweeks_to_process
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                gw = futures[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                        print(f"Completed processing for gameweek {gw}")
                    else:
                        print(f"Failed to process gameweek {gw}")
                except Exception as e:
                    print(f"Error processing gameweek {gw}: {e}")
    else:
        # Process gameweeks sequentially
        for gw in gameweeks_to_process:
            result = process_gameweek(
                gw,
                data_dir,
                base_output_dir,
                regression_models,
                classification_models,
                metadata,
                args.include_all_players,
                args.skip_explain,
                args.explain_global,
                args.explain_local,
                args.explain_method,
                args.max_features
            )
            
            if result:
                all_results.append(result)
    
    # Create summary files if requested and multiple gameweeks were processed
    if args.create_summary and len(all_results) > 1:
        create_summary_files(all_results, base_output_dir)
    
    print("\nAll gameweek processing complete!")

if __name__ == "__main__":
    main()
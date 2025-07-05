import argparse
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.pipeline import Pipeline

# Add argument parsing for explainability options
def parse_arguments():
    parser = argparse.ArgumentParser(description='Run predictions and generate explainability visualizations')
    parser.add_argument('--skip-explain', action='store_true', help='Skip generating explainability figures')
    parser.add_argument('--explain-global', action='store_true', help='Generate global explainability figures')
    parser.add_argument('--explain-local', action='store_true', help='Generate local explainability figures')
    parser.add_argument('--explain-method', choices=['shap', 'lime', 'both'], default='shap', 
                      help='Explainability method to use (default: shap)')
    parser.add_argument('--max-display', type=int, default=10, 
                      help='Maximum number of features to display in explainability charts')
    parser.add_argument('--output-dir', type=str, help='Custom output directory for explanations')
    
    return parser.parse_args()

# Helper function to create directories
def create_explanation_directories(output_dir: str, algorithm: str) -> Tuple[str, str, str]:
    """Create necessary directories for explainability outputs"""
    algo_dir = os.path.join(output_dir, algorithm)
    local_dir = os.path.join(algo_dir, "local_explanations")
    global_dir = os.path.join(algo_dir, "global_explanations")
    
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(global_dir, exist_ok=True)
    
    return algo_dir, local_dir, global_dir

# Function to get the appropriate SHAP explainer for a model
def get_shap_explainer(model, X_train, model_type: str):
    """
    Return the appropriate SHAP explainer for the given model type
    
    Args:
        model: The trained model
        X_train: Training data for explainer background
        model_type: Model type (e.g., 'RFReg', 'Ridge', 'DeepLearning', 'LSTM')
        
    Returns:
        SHAP explainer object
    """
    # Sample a smaller background dataset for efficiency
    if len(X_train) > 100:
        np.random.seed(42)
        background_indices = np.random.choice(len(X_train), 100, replace=False)
        background = X_train[background_indices]
    else:
        background = X_train
    
    # Choose the appropriate explainer based on model type
    if model_type in ['RFReg', 'GBReg', 'XGBoost'] or 'RandomForest' in str(type(model)):
        return shap.TreeExplainer(model)
    elif model_type in ['Ridge', 'LinearRegression'] or 'Ridge' in str(type(model)):
        return shap.LinearExplainer(model, background)
    elif model_type in ['DeepLearning', 'LSTM'] or 'keras' in str(type(model)).lower():
        # For neural networks, use DeepExplainer or GradientExplainer
        try:
            # Try DeepExplainer first
            return shap.DeepExplainer(model, background)
        except Exception as e:
            print(f"DeepExplainer failed: {e}. Trying GradientExplainer...")
            try:
                # If that fails, try GradientExplainer
                return shap.GradientExplainer(model, background)
            except Exception as e2:
                print(f"GradientExplainer failed: {e2}. Falling back to KernelExplainer...")
                # If all else fails, use KernelExplainer as a fallback
                predict_fn = lambda x: model.predict(x)
                return shap.KernelExplainer(predict_fn, background)
    else:
        # For any other model type, use KernelExplainer
        predict_fn = lambda x: model.predict(x)
        return shap.KernelExplainer(predict_fn, background)

# Function to create a LIME explainer
def get_lime_explainer(X_train, feature_names, categorical_features=None, discretize_continuous=True):
    """
    Create a LIME explainer for tabular data
    
    Args:
        X_train: Training data for explainer background
        feature_names: List of feature names
        categorical_features: List of indices for categorical features
        discretize_continuous: Whether to discretize continuous features
        
    Returns:
        LIME explainer object
    """
    if categorical_features is None:
        categorical_features = []
    
    return lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        categorical_features=categorical_features,
        discretize_continuous=discretize_continuous
    )

# Function to generate global SHAP explanations
def generate_global_shap_explanations(model, X, feature_names, model_type, output_path, position, max_display=10):
    """
    Generate global SHAP explanations and save visualizations
    
    Args:
        model: Trained model
        X: Feature data for generating explanations
        feature_names: List of feature names
        model_type: Type of model
        output_path: Directory to save visualizations
        position: Player position (GK, DEF, MID, FWD)
        max_display: Maximum number of features to display
    """
    print(f"Generating global SHAP explanations for {model_type} - {position}...")
    
    # Sample data if it's too large
    if len(X) > 100:
        np.random.seed(42)
        indices = np.random.choice(len(X), 100, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # Get the appropriate explainer
    explainer = get_shap_explainer(model, X, model_type)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # Handle different formats of SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # For models with multi-output
    
    # Create beeswarm plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X_sample, 
        feature_names=feature_names,
        max_display=max_display,
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
        X_sample,
        feature_names=feature_names,
        max_display=max_display,
        plot_type="bar",
        show=False
    )
    plt.title(f"{model_type} Mean Feature Impact - {position}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"global_bar_{position}.png"), dpi=150)
    plt.close()
    
    # Create feature dependence plots for top 5 features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(-mean_abs_shap)[:5]  # Top 5 features
    
    for idx in top_indices:
        if idx < len(feature_names):
            feature_name = feature_names[idx]
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                idx, 
                shap_values, 
                X_sample,
                feature_names=feature_names,
                show=False
            )
            plt.title(f"Dependence Plot for {feature_name} - {position}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"dependence_{position}_{feature_name.replace('/', '_')}.png"), dpi=150)
            plt.close()
    
    print(f"  -> Global SHAP explanations saved to {output_path}")

# Function to generate local SHAP explanations
def generate_local_shap_explanation(model, x, feature_names, model_type, output_path, 
                                   position, player_name, predicted_value, max_display=10):
    """
    Generate local SHAP explanation (waterfall plot) for a single prediction
    
    Args:
        model: Trained model
        x: Feature vector for the specific instance
        feature_names: List of feature names
        model_type: Type of model
        output_path: Directory to save visualizations
        position: Player position (GK, DEF, MID, FWD)
        player_name: Name of the player
        predicted_value: The predicted value
        max_display: Maximum number of features to display
    """
    print(f"Generating local SHAP explanation for {player_name} - {position}...")
    
    # Reshape input if needed
    if x.ndim == 1:
        x_reshaped = x.reshape(1, -1)
    else:
        x_reshaped = x
    
    # Special handling for LSTM models
    if model_type == 'LSTM':
        # For LSTM, we need to reshape the input to (samples, timesteps, features)
        # Assuming x is already scaled appropriately
        x_reshaped = np.expand_dims(x_reshaped, axis=1)  # Add timestep dimension
        predict_fn = lambda x: model.predict(x)
        explainer = shap.KernelExplainer(predict_fn, np.zeros((1, 1, x_reshaped.shape[2])))
        shap_values = explainer.shap_values(x_reshaped)
    else:
        # Get the appropriate explainer
        explainer = get_shap_explainer(model, x_reshaped, model_type)
        shap_values = explainer.shap_values(x_reshaped)
    
    # Handle different formats of SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    if shap_values.ndim > 1:
        shap_values = shap_values[0]
    
    # Get the base value (expected value)
    if hasattr(explainer, 'expected_value'):
        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[0]
    else:
        # If no expected_value, use mean of prediction as base
        base_value = predicted_value * 0.5
    
    # Create waterfall plot
    plt.figure(figsize=(12, 8))
    
    # Create SHAP explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=x.flatten(),
        feature_names=feature_names
    )
    
    # Generate waterfall plot showing how features contribute to the prediction
    shap.waterfall_plot(
        explanation, 
        max_display=max_display, 
        show=False
    )
    
    plt.title(f"{model_type} Prediction Explanation\n{player_name} - {position}\nPredicted Points: {predicted_value:.2f}")
    plt.tight_layout()
    
    # Clean player name for filename
    clean_name = player_name.replace('/', '_').replace(' ', '_').replace("'", "")
    plt.savefig(os.path.join(output_path, f"local_waterfall_{position}_{clean_name}.png"), dpi=150)
    plt.close()
    
    print(f"  -> Local SHAP explanation saved to {output_path}")

# Function to generate LIME explanations
def generate_lime_explanation(model, x, feature_names, model_type, output_path, 
                             position, player_name, predicted_value, X_train, num_features=10):
    """
    Generate LIME explanation for a single prediction
    
    Args:
        model: Trained model
        x: Feature vector for the specific instance
        feature_names: List of feature names
        model_type: Type of model
        output_path: Directory to save visualizations
        position: Player position (GK, DEF, MID, FWD)
        player_name: Name of the player
        predicted_value: The predicted value
        X_train: Training data for the LIME explainer
        num_features: Number of features to include in the explanation
    """
    print(f"Generating LIME explanation for {player_name} - {position}...")
    
    # Reshape input if needed
    if x.ndim == 1:
        x_reshaped = x.reshape(1, -1)
    else:
        x_reshaped = x
    
    # Create LIME explainer
    lime_explainer = get_lime_explainer(X_train, feature_names)
    
    # Define prediction function based on model type
    if model_type == 'LSTM':
        # For LSTM, we need to reshape the input
        def predict_fn(x):
            # Reshape to (samples, timesteps, features)
            x_3d = np.expand_dims(x, axis=1)
            return model.predict(x_3d)
    else:
        # Standard prediction function
        def predict_fn(x):
            return model.predict(x)
    
    # Generate explanation
    explanation = lime_explainer.explain_instance(
        x.flatten(),
        predict_fn,
        num_features=num_features
    )
    
    # Plot explanation
    plt.figure(figsize=(10, 6))
    explanation.as_pyplot_figure()
    plt.title(f"LIME Explanation - {player_name} ({position})\nPredicted Points: {predicted_value:.2f}")
    plt.tight_layout()
    
    # Clean player name for filename
    clean_name = player_name.replace('/', '_').replace(' ', '_').replace("'", "")
    plt.savefig(os.path.join(output_path, f"lime_explanation_{position}_{clean_name}.png"), dpi=150)
    plt.close()
    
    print(f"  -> LIME explanation saved to {output_path}")

# Main function to generate all explanations
def generate_model_explanations(
    regression_models, 
    df_pred, 
    drop_cols, 
    output_dir, 
    explain_global=True, 
    explain_local=True, 
    explain_method='shap', 
    max_display=10
):
    """
    Generate all explanations for regression models
    
    Args:
        regression_models: Dictionary of trained models
        df_pred: DataFrame with prediction data
        drop_cols: Columns to drop from feature data
        output_dir: Directory to save explanations
        explain_global: Whether to generate global explanations
        explain_local: Whether to generate local explanations
        explain_method: Explainability method ('shap', 'lime', or 'both')
        max_display: Maximum number of features to display in explanations
    """
    # Create a set of unique algorithms
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
            if explain_global and (explain_method in ['shap', 'both']):
                generate_global_shap_explanations(
                    model, X_reg_scaled, feature_names, algo_name, 
                    global_dir, pos_label, max_display
                )
            
            # Find the top player for local explanations
            if explain_local:
                # Make predictions
                if algo_name == 'LSTM' and 'lookback' in model_dict:
                    # For LSTM, reshape data appropriately
                    lookback = model_dict.get('lookback', 3)
                    
                    # Prepare LSTM data
                    X_lstm = np.zeros((X_reg_scaled.shape[0], lookback, X_reg_scaled.shape[1]))
                    for i in range(X_reg_scaled.shape[0]):
                        for t in range(lookback):
                            X_lstm[i, t, :] = X_reg_scaled[i]
                    
                    y_pred = model.predict(X_lstm).flatten()
                else:
                    # Standard prediction
                    y_pred = model.predict(X_reg_scaled).flatten()
                
                # Add predictions to dataframe
                df_pos['predicted_points'] = y_pred
                
                # Find top player
                top_idx = df_pos['predicted_points'].idxmax()
                top_player = df_pos.loc[top_idx]
                x_top = X_reg.loc[top_idx]
                x_top_scaled = X_reg_scaled[df_pos.index.get_loc(top_idx)]
                
                print(f"Top {pos_label} player: {top_player['name']} with predicted points: {top_player['predicted_points']:.2f}")
                
                # Generate local explanations
                if explain_method in ['shap', 'both']:
                    generate_local_shap_explanation(
                        model, x_top_scaled, feature_names, algo_name,
                        local_dir, pos_label, top_player['name'], 
                        top_player['predicted_points'], max_display
                    )
                
                if explain_method in ['lime', 'both']:
                    generate_lime_explanation(
                        model, x_top_scaled, feature_names, algo_name,
                        local_dir, pos_label, top_player['name'],
                        top_player['predicted_points'], X_reg_scaled, max_display
                    )

# Modified main script section
def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Load regression models and data
    script_dir = os.path.dirname(__file__)
    model_dir = os.path.join(script_dir, "..", "models")
    data_dir = os.path.join(script_dir, "..", "data")
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(script_dir, "..", "explanations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    with open(os.path.join(model_dir, "regression_models.pkl"), "rb") as f:
        regression_models = pickle.load(f)
    
    # Load prediction data
    prediction_data_path = os.path.join(data_dir, "player_predictions_data.csv")
    df_pred = pd.read_csv(prediction_data_path)
    
    # Add position_label column
    def get_position_label(row):
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
    
    df_pred['position_label'] = df_pred.apply(get_position_label, axis=1)
    df_pred = df_pred[df_pred['position_label'] != 'UNK']
    
    # Columns to drop for regression
    drop_cols = [
        'season_x', 'team_x', 'name', 'kickoff_time', 'opp_team_name', 'game_date',
        'next_match_points', 'played_next_match', 'started_next_match', 'next_fixture'
    ]
    
    # Check if explainability generation is requested
    if not args.skip_explain:
        generate_model_explanations(
            regression_models,
            df_pred,
            drop_cols,
            output_dir,
            explain_global=args.explain_global or not args.explain_local,
            explain_local=args.explain_local or not args.explain_global,
            explain_method=args.explain_method,
            max_display=args.max_display
        )
    else:
        print("Skipping explainability figure generation as requested.")
    
    print("\nExplanation generation complete!")

if __name__ == "__main__":
    main()
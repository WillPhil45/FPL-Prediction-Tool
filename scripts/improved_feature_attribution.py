import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from typing import Dict, List, Tuple, Any, Optional, Union

class ModelExplainer:
    """A class for explaining machine learning model predictions using SHAP and LIME"""
    
    def __init__(self, output_dir: str, method: str = 'shap', max_features: int = 10):
        """
        Initialize the explainer
        
        Args:
            output_dir: Directory to save explanation outputs
            method: Explainability method ('shap', 'lime', or 'both')
            max_features: Maximum number of features to display in explanations
        """
        self.output_dir = output_dir
        self.method = method
        self.max_features = max_features
        os.makedirs(output_dir, exist_ok=True)
    
    def create_directories(self, algorithm: str) -> Tuple[str, str, str]:
        """Create necessary directories for explanation outputs"""
        algo_dir = os.path.join(self.output_dir, algorithm)
        local_dir = os.path.join(algo_dir, "local_explanations")
        global_dir = os.path.join(algo_dir, "global_explanations")
        
        os.makedirs(local_dir, exist_ok=True)
        os.makedirs(global_dir, exist_ok=True)
        
        return algo_dir, local_dir, global_dir
    
    def get_shap_explainer(self, model, X_background, model_type: str):
        """Get the appropriate SHAP explainer for the model type"""
        # Sample a smaller dataset for background if needed
        if len(X_background) > 100:
            np.random.seed(42)
            background_indices = np.random.choice(len(X_background), 100, replace=False)
            background = X_background[background_indices]
        else:
            background = X_background
        
        # Choose explainer based on model type
        if model_type in ['RFReg', 'GBReg', 'XGBoost'] or 'RandomForest' in str(type(model)):
            return shap.TreeExplainer(model)
        elif model_type in ['Ridge', 'LinearRegression'] or 'linear' in str(type(model)).lower():
            return shap.LinearExplainer(model, background)
        elif model_type in ['DeepLearning', 'LSTM'] or 'keras' in str(type(model)).lower():
            # Try different neural network explainers
            try:
                return shap.DeepExplainer(model, background)
            except Exception as e:
                print(f"DeepExplainer failed: {e}. Trying GradientExplainer...")
                try:
                    return shap.GradientExplainer(model, background)
                except Exception as e2:
                    print(f"GradientExplainer failed: {e2}. Using KernelExplainer as fallback.")
                    return shap.KernelExplainer(model.predict, background)
        else:
            # Fallback to KernelExplainer for any other model
            return shap.KernelExplainer(model.predict, background)
    
    def explain_global(self, model, X, feature_names, algo_name, pos_label):
        """Generate global explanations for a model"""
        print(f"Generating global explanations for {algo_name} - {pos_label}...")
        
        _, _, global_dir = self.create_directories(algo_name)
        
        # Sample data if needed
        if len(X) > 100:
            np.random.seed(42)
            indices = np.random.choice(len(X), 100, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # SHAP explanations
        if self.method in ['shap', 'both']:
            self._generate_shap_global(model, X_sample, feature_names, algo_name, pos_label, global_dir)
        
        # LIME explanations (global feature importance)
        if self.method in ['lime', 'both']:
            self._generate_lime_global(model, X_sample, feature_names, algo_name, pos_label, global_dir)
    
    def _generate_shap_global(self, model, X, feature_names, algo_name, pos_label, output_dir):
        """Generate global SHAP explanations"""
        # Get explainer and SHAP values
        explainer = self.get_shap_explainer(model, X, algo_name)
        shap_values = explainer.shap_values(X)
        
        # Handle different shapes of SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Create beeswarm plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=feature_names,
            max_display=self.max_features,
            show=False
        )
        plt.title(f"{algo_name} Global Feature Importance - {pos_label}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"global_beeswarm_{pos_label}.png"), dpi=150)
        plt.close()
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X,
            feature_names=feature_names,
            max_display=self.max_features,
            plot_type="bar",
            show=False
        )
        plt.title(f"{algo_name} Mean Feature Impact - {pos_label}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"global_bar_{pos_label}.png"), dpi=150)
        plt.close()
        
        # Create heatmap for feature interactions
        try:
            if X.shape[1] <= 20:  # Only for datasets with reasonable feature counts
                plt.figure(figsize=(12, 10))
                shap.plots.heatmap(shap.Explanation(
                    values=shap_values,
                    data=X,
                    feature_names=feature_names
                ), max_display=10, show=False)
                plt.title(f"{algo_name} Feature Interaction Heatmap - {pos_label}")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"interaction_heatmap_{pos_label}.png"), dpi=150)
                plt.close()
        except Exception as e:
            print(f"Could not generate interaction heatmap: {e}")
        
        print(f"  -> Global SHAP explanations saved to {output_dir}")
    
    def _generate_lime_global(self, model, X, feature_names, algo_name, pos_label, output_dir):
        """Generate global LIME feature importance"""
        # Create LIME explainer
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            feature_names=feature_names,
            discretize_continuous=True
        )
        
        # Get feature importances across multiple samples
        feature_importances = np.zeros(len(feature_names))
        num_samples = min(20, len(X))  # Limit to 20 samples for efficiency
        
        for i in range(num_samples):
            explanation = lime_explainer.explain_instance(
                X[i], 
                model.predict, 
                num_features=len(feature_names)
            )
            
            # Extract feature importances
            for feature, importance in explanation.as_list():
                idx = feature_names.index(feature.split(" ")[0])
                feature_importances[idx] += abs(importance)
        
        # Average and normalize
        feature_importances /= num_samples
        
        # Sort features by importance
        indices = np.argsort(feature_importances)[::-1][:self.max_features]
        top_features = [feature_names[i] for i in indices]
        top_importances = [feature_importances[i] for i in indices]
        
        # Plot bar chart
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_importances, align='center')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Average Absolute Importance')
        plt.title(f"{algo_name} Global LIME Feature Importance - {pos_label}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"lime_global_importance_{pos_label}.png"), dpi=150)
        plt.close()
        
        print(f"  -> Global LIME explanations saved to {output_dir}")
    
    def explain_local(self, model, x, feature_names, algo_name, pos_label, player_name, predicted_value, X_background=None):
        """Generate local explanations for a single prediction"""
        print(f"Generating local explanations for {player_name} - {pos_label}...")
        
        _, local_dir, _ = self.create_directories(algo_name)
        
        # Clean player name for filenames
        clean_name = player_name.replace('/', '_').replace(' ', '_').replace("'", "")
        
        # Reshape input if needed
        if x.ndim == 1:
            x_reshaped = x.reshape(1, -1)
        else:
            x_reshaped = x
        
        # SHAP explanations
        if self.method in ['shap', 'both']:
            self._generate_shap_local(model, x_reshaped, X_background, feature_names, 
                                     algo_name, pos_label, player_name, predicted_value, local_dir, clean_name)
        
        # LIME explanations
        if self.method in ['lime', 'both'] and X_background is not None:
            self._generate_lime_local(model, x_reshaped[0], X_background, feature_names,
                                     algo_name, pos_label, player_name, predicted_value, local_dir, clean_name)
    
    def _generate_shap_local(self, model, x, X_background, feature_names, 
                           algo_name, pos_label, player_name, predicted_value, output_dir, clean_name):
        """Generate local SHAP explanation for a single prediction"""
        # Get explainer
        explainer = self.get_shap_explainer(model, X_background if X_background is not None else x, algo_name)
        
        # Get SHAP values
        shap_values = explainer.shap_values(x)
        
        # Handle different formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        if shap_values.ndim > 1 and shap_values.shape[0] == 1:
            shap_values = shap_values[0]
        
        # Get base value
        if hasattr(explainer, 'expected_value'):
            base_value = explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[0]
        else:
            base_value = predicted_value * 0.5
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=x.flatten() if x.ndim > 1 else x,
            feature_names=feature_names
        )
        
        shap.waterfall_plot(explanation, max_display=self.max_features, show=False)
        plt.title(f"{algo_name} Prediction Explanation\n{player_name} - {pos_label}\nPredicted Points: {predicted_value:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_waterfall_{pos_label}_{clean_name}.png"), dpi=150)
        plt.close()
        
        # Create force plot as well
        try:
            plt.figure(figsize=(12, 3))
            force_plot = shap.force_plot(
                base_value,
                shap_values,
                x.flatten() if x.ndim > 1 else x,
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            plt.title(f"{algo_name} Force Plot\n{player_name} - {pos_label}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_force_{pos_label}_{clean_name}.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Could not generate force plot: {e}")
        
        print(f"  -> Local SHAP explanations saved to {output_dir}")
    
    def _generate_lime_local(self, model, x, X_background, feature_names,
                           algo_name, pos_label, player_name, predicted_value, output_dir, clean_name):
        """Generate local LIME explanation for a single prediction"""
        # Create LIME explainer
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_background,
            feature_names=feature_names,
            discretize_continuous=True
        )
        
        # Get explanation
        explanation = lime_explainer.explain_instance(
            x,
            model.predict,
            num_features=self.max_features
        )
        
        # Plot explanation
        plt.figure(figsize=(10, 6))
        explanation.as_pyplot_figure()
        plt.title(f"LIME Explanation - {player_name} ({pos_label})\nPredicted Points: {predicted_value:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"lime_explanation_{pos_label}_{clean_name}.png"), dpi=150)
        plt.close()
        
        print(f"  -> Local LIME explanation saved to {output_dir}")
    
    def explain_regression_models(self, regression_models, df_pred, drop_cols):
        """Generate explanations for all regression models"""
        # Get unique algorithms
        algos = set(algo for (pos_label, algo) in regression_models.keys())
        
        for algo_name in algos:
            print(f"\nProcessing explanations for {algo_name} models...")
            
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
                
                # Generate global explanations
                self.explain_global(model, X_reg_scaled, feature_names, algo_name, pos_label)
                
                # Make predictions for local explanations
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
                x_top_scaled = X_reg_scaled[df_pos.index.get_loc(top_idx)]
                
                print(f"Top {pos_label} player: {top_player['name']} with predicted points: {top_player['predicted_points']:.2f}")
                
                # Generate local explanation for top player
                self.explain_local(
                    model, x_top_scaled, feature_names, algo_name, pos_label,
                    top_player['name'], top_player['predicted_points'], X_reg_scaled
                )

# Function to handle LSTM models specifically
def explain_lstm_model(model, X, feature_names, model_dict, algo_name, pos_label, player_name=None, output_dir=None):
    """
    Generate explanations for LSTM models using appropriate reshaping
    
    Args:
        model: LSTM model
        X: Feature data (unreshaped)
        feature_names: List of feature names
        model_dict: Dictionary containing model metadata
        algo_name: Algorithm name
        pos_label: Position label
        player_name: Player name (for local explanations)
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = 'explanations'
    
    # Get lookback window size
    lookback = model_dict.get('lookback', 3)
    
    # Reshape data for LSTM
    if X.ndim == 2:
        # Multiple samples
        X_lstm = np.zeros((X.shape[0], lookback, X.shape[1]))
        for i in range(X.shape[0]):
            for t in range(lookback):
                X_lstm[i, t, :] = X[i]
    else:
        # Single sample
        X_lstm = np.zeros((1, lookback, X.shape[0]))
        for t in range(lookback):
            X_lstm[0, t, :] = X
    
    # Create surrogate model for explanation
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Get predictions from LSTM
    lstm_preds = model.predict(X_lstm).flatten()
    
    # For global explanation, train surrogate on sample
    if X.ndim == 2:
        surrogate = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        surrogate.fit(X, lstm_preds)
        
        # Use SHAP on surrogate
        explainer = shap.TreeExplainer(surrogate)
        shap_values = explainer.shap_values(X)
        
        # Create global importance plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.title(f"LSTM Global Feature Importance (Surrogate) - {pos_label}")
        plt.tight_layout()
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"lstm_global_{pos_label}.png"), dpi=150)
        plt.close()
    
    # For local explanation (if player_name provided)
    if player_name is not None:
        # Train surrogate on perturbed samples around this point
        X_perturbed = []
        y_perturbed = []
        
        # Create perturbed samples
        np.random.seed(42)
        for _ in range(1000):
            # Add random noise
            noise = np.random.normal(0, 0.1, size=X.shape)
            x_new = X + noise
            
            # Reshape for LSTM
            x_lstm = np.zeros((1, lookback, X.shape[0]))
            for t in range(lookback):
                x_lstm[0, t, :] = x_new
            
            # Get prediction
            pred = model.predict(x_lstm)[0][0]
            
            # Store
            X_perturbed.append(x_new)
            y_perturbed.append(pred)
        
        # Convert to arrays
        X_perturbed = np.array(X_perturbed)
        y_perturbed = np.array(y_perturbed)
        
        # Train surrogate
        surrogate = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        surrogate.fit(X_perturbed, y_perturbed)
        
        # Use SHAP on surrogate for local explanation
        explainer = shap.TreeExplainer(surrogate)
        shap_values = explainer.shap_values(X.reshape(1, -1))
        base_value = explainer.expected_value
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=X,
            feature_names=feature_names
        )
        
        shap.waterfall_plot(explanation, max_display=10, show=False)
        plt.title(f"LSTM Prediction Explanation (Surrogate)\n{player_name} - {pos_label}")
        plt.tight_layout()
        
        # Clean player name for filename
        clean_name = player_name.replace('/', '_').replace(' ', '_').replace("'", "")
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"lstm_local_{pos_label}_{clean_name}.png"), dpi=150)
        plt.close()
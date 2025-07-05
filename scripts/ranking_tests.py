#!/usr/bin/env python3
"""
FPL Model Ranking Evaluation

This script evaluates how well different prediction models rank FPL players
by calculating Kendall's Tau-b and nDCG metrics for each model, position, and gameweek.

It handles the FPL's unique structure where gameweek X predictions are for gameweek X+1,
and produces both detailed and summary statistics across all available data.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score
import glob
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from tqdm import tqdm
import argparse

def get_actual_results(gw_number, base_path="Up-to-data/Fantasy-Premier-League/data/2024-25/gws"):
    """
    Load actual results for a specific gameweek
    
    Parameters:
    - gw_number: The gameweek number to load
    - base_path: Path to directory containing gameweek result files
    
    Returns:
    - DataFrame with player results or None if file not found
    """
    file_path = os.path.join(base_path, f"gw{gw_number}.csv")
    
    if not os.path.exists(file_path):
        print(f"Warning: Actual results file not found for GW{gw_number}: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # Check for required columns
        required_columns = ['name', 'position', 'total_points']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            print(f"Error: Missing columns in actual results file for GW{gw_number}: {missing}")
            return None
        
        # Keep only necessary columns
        df = df[required_columns]
        # Rename 'position' to 'position_label' for consistency
        df = df.rename(columns={'position': 'position_label'})
        
        # Handle duplicate player names if they exist
        if df['name'].duplicated().any():
            print(f"Warning: Found duplicate player names in GW{gw_number} results. Using first occurrence.")
            df = df.drop_duplicates(subset=['name'], keep='first')
        
        return df
    except Exception as e:
        print(f"Error loading actual results for GW{gw_number}: {e}")
        return None

def get_predictions(model_name, gw_number, predictions_path="predictions"):
    """
    Load predictions from a specific model for a gameweek
    
    Parameters:
    - model_name: Name of the model technique (e.g., "DeepLearning")
    - gw_number: Gameweek number for predictions
    - predictions_path: Base path to predictions directory
    
    Returns:
    - DataFrame with model predictions or None if file not found
    """
    file_path = os.path.join(predictions_path, f"gw{gw_number:02d}", 
                            model_name, "predictions", 
                            f"predictions_{model_name}_regression.csv")
    
    if not os.path.exists(file_path):
        print(f"Warning: Predictions file not found for {model_name} GW{gw_number}: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # Ensure we have the necessary columns
        required_columns = ['name', 'position_label', 'predicted_points']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            print(f"Error: Missing columns in predictions file for {model_name} GW{gw_number}: {missing}")
            return None
        
        # Keep only necessary columns
        df = df[required_columns]
        return df
    except Exception as e:
        print(f"Error loading predictions for {model_name} GW{gw_number}: {e}")
        return None

def calculate_metrics(actual_df, pred_df, verbose=False):
    """
    Calculate Kendall's Tau-b and nDCG for each position
    
    Parameters:
    - actual_df: DataFrame with actual player results
    - pred_df: DataFrame with model predictions
    - verbose: Whether to print detailed matching information
    
    Returns:
    - Dictionary with metrics by position
    """
    # Merge actual and predicted data
    merged_df = pd.merge(actual_df, pred_df, on=['name', 'position_label'])
    
    if verbose:
        print(f"Matched {len(merged_df)} players out of {len(pred_df)} predictions and {len(actual_df)} actuals")
    
    results = {}
    position_labels = ['GK', 'DEF', 'MID', 'FWD']
    
    for position in position_labels:
        pos_df = merged_df[merged_df['position_label'] == position]
        
        if len(pos_df) < 2:
            # Need at least 2 players to calculate correlation
            results[position] = {'kendall_tau': np.nan, 'ndcg': np.nan, 'count': len(pos_df)}
            if verbose:
                print(f"Position {position}: Not enough players ({len(pos_df)}) to calculate metrics")
            continue
        
        # Calculate Kendall's Tau-b (handles ties in both datasets)
        tau, p_value = kendalltau(pos_df['total_points'], pos_df['predicted_points'])
        
        # Calculate nDCG (normalized discounted cumulative gain)
        # Handle negative values in actual scores by shifting all values to ensure non-negative
        actual_points = pos_df['total_points'].values
        min_points = min(actual_points)
        
        # If there are negative values, shift all points to make them non-negative
        if min_points < 0:
            shifted_points = actual_points - min_points  # Shift by minimum to ensure all ≥ 0
        else:
            shifted_points = actual_points
            
        # For nDCG, we need to provide the scores in a specific format
        actual_scores = np.array([shifted_points])
        pred_scores = np.array([pos_df['predicted_points'].values])
        
        try:
            ndcg = ndcg_score(actual_scores, pred_scores)
        except Exception as e:
            print(f"Error calculating nDCG for {position}: {e}")
            ndcg = np.nan
        
        results[position] = {
            'kendall_tau': tau,
            'ndcg': ndcg,
            'count': len(pos_df),
            'p_value': p_value
        }
        
        if verbose:
            print(f"Position {position} ({len(pos_df)} players): Kendall's Tau = {tau:.4f}, nDCG = {ndcg:.4f}")
    
    # Calculate overall metrics (across all positions)
    if len(merged_df) >= 2:
        tau, p_value = kendalltau(merged_df['total_points'], merged_df['predicted_points'])
        
        # Handle negative values for overall nDCG
        actual_points = merged_df['total_points'].values
        min_points = min(actual_points)
        
        # If there are negative values, shift all points to make them non-negative
        if min_points < 0:
            shifted_points = actual_points - min_points  # Shift by minimum to ensure all ≥ 0
        else:
            shifted_points = actual_points
            
        actual_scores = np.array([shifted_points])
        pred_scores = np.array([merged_df['predicted_points'].values])
        
        try:
            ndcg = ndcg_score(actual_scores, pred_scores)
        except Exception as e:
            print(f"Error calculating overall nDCG: {e}")
            ndcg = np.nan
        
        results['ALL'] = {
            'kendall_tau': tau,
            'ndcg': ndcg,
            'count': len(merged_df),
            'p_value': p_value
        }
        
        if verbose:
            print(f"Overall ({len(merged_df)} players): Kendall's Tau = {tau:.4f}, nDCG = {ndcg:.4f}")
    else:
        results['ALL'] = {'kendall_tau': np.nan, 'ndcg': np.nan, 'count': len(merged_df), 'p_value': np.nan}
    
    return results

def print_results(all_results):
    """
    Print a summary of the evaluation results
    
    Parameters:
    - all_results: Dictionary containing results for all models
    """
    print("\n===== MODEL EVALUATION SUMMARY =====")
    
    for model_name, model_results in all_results.items():
        print(f"\n--- {model_name} ---")
        
        print("Average Metrics by Position:")
        
        # Print header
        print(f"{'Position':<6} | {'Kendall Tau-b':<13} | {'nDCG':<8} | {'GWs'}")
        print("-" * 40)
        
        # Print position results
        for position in ['GK', 'DEF', 'MID', 'FWD', 'ALL']:
            if position in model_results['average']:
                pos_results = model_results['average'][position]
                kt = pos_results['kendall_tau']
                ndcg = pos_results['ndcg']
                gws = pos_results['valid_gws']
                
                kt_str = f"{kt:.4f}" if not np.isnan(kt) else "N/A"
                ndcg_str = f"{ndcg:.4f}" if not np.isnan(ndcg) else "N/A"
                
                print(f"{position:<6} | {kt_str:<13} | {ndcg_str:<8} | {gws}")
        
        print()

def save_results_to_csv(all_results, output_dir="model_evaluation_results"):
    """
    Save detailed results to CSV files
    
    Parameters:
    - all_results: Dictionary containing results for all models
    - output_dir: Directory to save results
    """
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save overall summary
    summary_rows = []
    
    for model_name, model_results in all_results.items():
        for position, metrics in model_results['average'].items():
            summary_rows.append({
                'model': model_name,
                'position': position,
                'kendall_tau': metrics['kendall_tau'],
                'ndcg': metrics['ndcg'],
                'valid_gameweeks': metrics['valid_gws']
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "model_ranking_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary results to {summary_path}")
    
    # Save detailed gameweek results
    detailed_rows = []
    
    for model_name, model_results in all_results.items():
        for gw, gw_metrics in model_results['gameweeks'].items():
            for position, pos_metrics in gw_metrics.items():
                # Add p-value if available
                p_value = pos_metrics.get('p_value', np.nan)
                
                detailed_rows.append({
                    'model': model_name,
                    'gameweek': gw,
                    'target_gameweek': gw + 1,  # The gameweek being predicted
                    'position': position,
                    'kendall_tau': pos_metrics['kendall_tau'],
                    'ndcg': pos_metrics['ndcg'],
                    'player_count': pos_metrics['count'],
                    'p_value': p_value
                })
    
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_path = os.path.join(output_dir, "model_ranking_detailed.csv")
    detailed_df.to_csv(detailed_path, index=False)
    print(f"Saved detailed results to {detailed_path}")

def create_visualizations(all_results, output_dir="model_evaluation_results"):
    """
    Create visualizations of the evaluation results
    
    Parameters:
    - all_results: Dictionary containing results for all models
    - output_dir: Directory to save visualizations
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set the style for all plots
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Prepare data for plots
    models = list(all_results.keys())
    positions = ['GK', 'DEF', 'MID', 'FWD','ALL']
    
    # Use a colorblind-friendly palette
    colors = sns.color_palette("colorblind", n_colors=len(models))
    
    # Plot Kendall's Tau-b comparison
    plt.figure(figsize=(11, 6))
    
    bar_width = 0.15
    index = np.arange(len(positions))
    
    for i, model_name in enumerate(models):
        kt_values = []
        
        for position in positions:
            if position in all_results[model_name]['average']:
                kt_values.append(all_results[model_name]['average'][position]['kendall_tau'])
            else:
                kt_values.append(np.nan)
        
        plt.bar(index + i*bar_width, kt_values, bar_width, 
                label=model_name, color=colors[i], alpha=0.8)
    
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Kendall\'s Tau-b', fontsize=12)
    plt.title('Player Ranking Accuracy by Model and Position\n(Kendall\'s Tau-b)', fontsize=14)
    plt.xticks(index + bar_width * (len(models)-1) / 2, positions, fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, model_name in enumerate(models):
        for j, position in enumerate(positions):
            if position in all_results[model_name]['average']:
                value = all_results[model_name]['average'][position]['kendall_tau']
                if not np.isnan(value):
                    plt.text(j + i*bar_width, value + 0.01, 
                             f"{value:.3f}", ha='center', va='bottom', 
                             fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "kendall_tau_comparison.png"), dpi=300)
    plt.close()
    
    # Plot nDCG comparison
    plt.figure(figsize=(14, 9))
    
    for i, model_name in enumerate(models):
        ndcg_values = []
        
        for position in positions:
            if position in all_results[model_name]['average']:
                ndcg_values.append(all_results[model_name]['average'][position]['ndcg'])
            else:
                ndcg_values.append(np.nan)
        
        plt.bar(index + i*bar_width, ndcg_values, bar_width, 
                label=model_name, color=colors[i], alpha=0.8)
    
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('nDCG', fontsize=12)
    plt.title('Top Player Identification Accuracy by Model and Position\n(nDCG Score)', fontsize=14)
    plt.xticks(index + bar_width * (len(models)-1) / 2, positions, fontsize=11)
    plt.legend(fontsize=10, loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, model_name in enumerate(models):
        for j, position in enumerate(positions):
            if position in all_results[model_name]['average']:
                value = all_results[model_name]['average'][position]['ndcg']
                if not np.isnan(value):
                    plt.text(j + i*bar_width, value + 0.01, 
                             f"{value:.3f}", ha='center', va='bottom', 
                             fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "ndcg_comparison.png"), dpi=300)
    plt.close()
    
    # Create time series plots to show metric trends over gameweeks
    create_trend_visualizations(all_results, plots_dir)
    
    print(f"Saved visualizations to {plots_dir}")

def create_trend_visualizations(all_results, plots_dir):
    """
    Create time series visualizations showing trends over gameweeks
    
    Parameters:
    - all_results: Dictionary containing results for all models
    - plots_dir: Directory to save visualizations
    """
    positions = ['GK', 'DEF', 'MID', 'FWD', 'ALL']
    
    # For each position, create a trend plot for both metrics
    for position in positions:
        # Get all gameweeks that have data for this position
        gameweeks = set()
        for model_results in all_results.values():
            for gw in model_results['gameweeks'].keys():
                if (position in model_results['gameweeks'][gw] and 
                    not np.isnan(model_results['gameweeks'][gw][position]['kendall_tau'])):
                    gameweeks.add(gw)
        
        if not gameweeks:
            continue
            
        gameweeks = sorted(gameweeks)
        
        # Plot Kendall's Tau-b trend
        plt.figure(figsize=(14, 8))
        
        for model_name, model_results in all_results.items():
            gws = []
            taus = []
            
            for gw in gameweeks:
                if (gw in model_results['gameweeks'] and 
                    position in model_results['gameweeks'][gw] and 
                    not np.isnan(model_results['gameweeks'][gw][position]['kendall_tau'])):
                    gws.append(gw)
                    taus.append(model_results['gameweeks'][gw][position]['kendall_tau'])
            
            if gws:
                plt.plot(gws, taus, 'o-', linewidth=2, markersize=8, alpha=0.7, label=model_name)
        
        plt.xlabel('Prediction Gameweek', fontsize=12)
        plt.ylabel('Kendall\'s Tau-b', fontsize=12)
        plt.title(f'{position} Position: Ranking Accuracy Over Time (Kendall\'s Tau-b)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # Set x-axis to show only integer gameweeks
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add target gameweek annotation
        plt.figtext(0.5, 0.01, "Note: Prediction GW X is forecasting for GW X+1", 
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust to make room for the note
        plt.savefig(os.path.join(plots_dir, f"kendall_tau_trend_{position}.png"), dpi=300)
        plt.close()
        
        # Plot nDCG trend
        plt.figure(figsize=(14, 8))
        
        for model_name, model_results in all_results.items():
            gws = []
            ndcgs = []
            
            for gw in gameweeks:
                if (gw in model_results['gameweeks'] and 
                    position in model_results['gameweeks'][gw] and 
                    not np.isnan(model_results['gameweeks'][gw][position]['ndcg'])):
                    gws.append(gw)
                    ndcgs.append(model_results['gameweeks'][gw][position]['ndcg'])
            
            if gws:
                plt.plot(gws, ndcgs, 'o-', linewidth=2, markersize=8, alpha=0.7, label=model_name)
        
        plt.xlabel('Prediction Gameweek', fontsize=12)
        plt.ylabel('nDCG', fontsize=12)
        plt.title(f'{position} Position: Top Player Identification Over Time (nDCG)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # Set x-axis to show only integer gameweeks
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add target gameweek annotation
        plt.figtext(0.5, 0.01, "Note: Prediction GW X is forecasting for GW X+1", 
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust to make room for the note
        plt.savefig(os.path.join(plots_dir, f"ndcg_trend_{position}.png"), dpi=300)
        plt.close()

def evaluate_all_models(predictions_path="predictions", results_path="Up-to-data/Fantasy-Premier-League/data/2024-25/gws", 
                        output_dir="model_evaluation_results", verbose=False):
    """
    Evaluate all models across all available gameweeks
    
    Parameters:
    - predictions_path: Path to predictions directory
    - results_path: Path to actual results directory
    - output_dir: Directory to save results and visualizations
    - verbose: Whether to print detailed information during processing
    """
    # Find all available gameweeks
    prediction_dirs = glob.glob(os.path.join(predictions_path, "gw*"))
    gameweeks = []
    for dir_path in prediction_dirs:
        match = re.search(r'gw(\d+)', dir_path)
        if match:
            gameweeks.append(int(match.group(1)))
    
    gameweeks = sorted(gameweeks)
    
    if not gameweeks:
        print("No gameweek predictions found.")
        return
    
    print(f"Found predictions for gameweeks: {gameweeks}")
    
    # Find all model techniques (regression models only)
    model_techniques = []
    for gw in gameweeks:
        model_dirs = glob.glob(os.path.join(predictions_path, f"gw{gw:02d}/*"))
        for dir_path in model_dirs:
            model_name = os.path.basename(dir_path)
            # Skip classification models
            if "Clf" in model_name:
                continue
            
            # Check if it's a regression model directory (has 'predictions' subdirectory)
            pred_dir = os.path.join(dir_path, "predictions")
            pred_file = os.path.join(pred_dir, f"predictions_{model_name}_regression.csv")
            
            if os.path.isdir(pred_dir) and os.path.exists(pred_file) and model_name not in model_techniques:
                model_techniques.append(model_name)
    
    print(f"Found regression model techniques: {model_techniques}")
    
    # Initialize results structure
    all_results = {}
    
    # Process each model and gameweek
    for model_name in model_techniques:
        print(f"\nProcessing {model_name} model...")
        
        model_results = {
            'gameweeks': {},
            'average': {}
        }
        
        valid_gw_count = 0
        position_metrics = {
            'GK': {'kendall_tau': [], 'ndcg': []},
            'DEF': {'kendall_tau': [], 'ndcg': []},
            'MID': {'kendall_tau': [], 'ndcg': []},
            'FWD': {'kendall_tau': [], 'ndcg': []},
            'ALL': {'kendall_tau': [], 'ndcg': []}
        }
        
        # Use tqdm for progress bar if processing multiple gameweeks
        gw_iterator = tqdm(gameweeks) if len(gameweeks) > 1 else gameweeks
        
        for gw in gw_iterator:
            # GW1 predictions are for GW2, GW2 predictions are for GW3, etc.
            next_gw = gw + 1
            
            # Get actual results
            actual_df = get_actual_results(next_gw, results_path)
            if actual_df is None:
                if verbose:
                    print(f"Skipping GW{gw} for {model_name} (no actual results for GW{next_gw})")
                continue
            
            # Get predictions
            pred_df = get_predictions(model_name, gw, predictions_path)
            if pred_df is None:
                continue
            
            try:
                # Calculate metrics
                if verbose:
                    print(f"\nEvaluating {model_name} GW{gw} (predicting GW{next_gw}):")
                
                metrics = calculate_metrics(actual_df, pred_df, verbose)
                model_results['gameweeks'][gw] = metrics
                
                # Collect metrics for averaging
                valid_gw_count += 1
                for position in position_metrics.keys():
                    if position in metrics:
                        position_metrics[position]['kendall_tau'].append(metrics[position]['kendall_tau'])
                        position_metrics[position]['ndcg'].append(metrics[position]['ndcg'])
            except Exception as e:
                print(f"Error evaluating {model_name} GW{gw}: {e}")
                continue
        
        # Calculate averages
        if valid_gw_count > 0:
            for position in position_metrics.keys():
                kendall_values = [v for v in position_metrics[position]['kendall_tau'] if not np.isnan(v)]
                ndcg_values = [v for v in position_metrics[position]['ndcg'] if not np.isnan(v)]
                
                model_results['average'][position] = {
                    'kendall_tau': np.mean(kendall_values) if kendall_values else np.nan,
                    'ndcg': np.mean(ndcg_values) if ndcg_values else np.nan,
                    'valid_gws': len(kendall_values)
                }
        
        all_results[model_name] = model_results
        
        if verbose:
            print(f"\nProcessed {valid_gw_count} valid gameweeks for {model_name}")
    
    # Print results
    print_results(all_results)
    
    # Save results to CSV
    save_results_to_csv(all_results, output_dir)
    
    # Create visualizations
    create_visualizations(all_results, output_dir)
    
    return all_results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate FPL prediction model ranking accuracy')
    
    parser.add_argument('--predictions-path', type=str, default="predictions",
                        help='Path to predictions directory (default: predictions)')
    
    parser.add_argument('--results-path', type=str, 
                        default="Up-to-data/Fantasy-Premier-League/data/2024-25/gws",
                        help='Path to actual results directory')
    
    parser.add_argument('--output-dir', type=str, default="model_evaluation_results",
                        help='Directory to save results and visualizations')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information during processing')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    evaluate_all_models(
        predictions_path=args.predictions_path,
        results_path=args.results_path,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
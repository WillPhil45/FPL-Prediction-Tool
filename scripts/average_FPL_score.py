import os
import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse

def find_predicted_teams(predictions_base_path: str) -> List[Dict]:
    """
    Find all predicted team files across gameweeks and techniques
    
    Args:
        predictions_base_path: Path to the predictions directory
        
    Returns:
        List of dictionaries containing details of predicted team files
    """
    team_files = []
    
    # Pattern to match predicted team files
    pattern = os.path.join(predictions_base_path, "gw*", "*", "best_team", "predicted_team_*_gw*.csv")
    
    for filepath in glob.glob(pattern):
        # Extract gameweek from filepath
        gw_match = re.search(r'gw(\d+)', filepath)
        if not gw_match:
            continue
            
        gw = int(gw_match.group(1))
        
        # Extract algorithm/technique from filepath
        algo_match = re.search(r'predicted_team_([^_]+)_gw', filepath)
        if not algo_match:
            continue
            
        team_files.append({
            'gameweek': gw,
            'algorithm': algo_match.group(1),
            'filepath': filepath,
            'target_gameweek': gw + 1  # The gameweek we'll check actual points for
        })
    
    return team_files

def load_average_points(avg_points_path: str) -> Dict[int, float]:
    """
    Load average points per gameweek
    
    Args:
        avg_points_path: Path to the average points CSV
        
    Returns:
        Dictionary mapping gameweeks to average points
    """
    try:
        avg_df = pd.read_csv(avg_points_path)
        # Convert to dictionary for easy lookup
        return dict(zip(avg_df['gw'], avg_df['avg_points']))
    except Exception as e:
        print(f"Error loading average points data: {e}")
        return {}

def get_actual_team_points(predicted_team_df: pd.DataFrame, actual_data_path: str) -> Tuple[pd.DataFrame, float, str, float]:
    """
    Calculate actual points earned by the predicted team, with captaincy simulation
    
    Args:
        predicted_team_df: DataFrame with the predicted team
        actual_data_path: Path to the actual gameweek data
        
    Returns:
        Tuple of (enriched DataFrame with actual points, total points with captaincy, captain name, 
                 total points without captaincy)
    """
    try:
        # Load actual gameweek data
        actual_df = pd.read_csv(actual_data_path)
        
        # Ensure the name columns are clean and comparable
        predicted_team_df['name_cleaned'] = predicted_team_df['name'].str.strip().str.lower()
        actual_df['name_cleaned'] = actual_df['name'].str.strip().str.lower()
        
        # Create a lookup dictionary for player points
        actual_points_dict = dict(zip(actual_df['name_cleaned'], actual_df['total_points']))
        
        # Add actual points to predicted team dataframe
        predicted_team_df['actual_points'] = predicted_team_df['name_cleaned'].map(actual_points_dict)
        
        # Handle missing players (those who didn't play)
        missing_players = predicted_team_df[predicted_team_df['actual_points'].isna()]
        if not missing_players.empty:
            print(f"Warning: {len(missing_players)} players not found in actual data:")
            for _, player in missing_players.iterrows():
                print(f"  - {player['name']}")
            
            # Fill missing values with 0
            predicted_team_df['actual_points'] = predicted_team_df['actual_points'].fillna(0)
        
        # Calculate total points without captaincy (for prediction accuracy comparison)
        total_points_no_captaincy = predicted_team_df['actual_points'].sum()
        
        # Simulate captaincy - select player with highest predicted points as captain
        captain_idx = predicted_team_df['predicted_points'].idxmax()
        captain = predicted_team_df.loc[captain_idx]
        captain_name = captain['name']
        
        # Add captain marker to the dataframe
        predicted_team_df['is_captain'] = False
        predicted_team_df.loc[captain_idx, 'is_captain'] = True
        
        # Add captain_points column (same as actual_points for regular players, doubled for captain)
        predicted_team_df['captain_points'] = predicted_team_df['actual_points']
        predicted_team_df.loc[captain_idx, 'captain_points'] = predicted_team_df.loc[captain_idx, 'actual_points'] * 2
        
        # Calculate total points with captaincy
        captain_bonus = predicted_team_df.loc[captain_idx, 'actual_points']  # The extra points from captain
        total_points_with_captaincy = predicted_team_df['captain_points'].sum()
        
        # Log the captain selection and points
        print(f"Captain selected: {captain_name} (predicted: {captain['predicted_points']:.2f}, actual: {captain['actual_points']:.1f}, captaincy bonus: +{captain_bonus:.1f})")
        
        return predicted_team_df, total_points_with_captaincy, captain_name, total_points_no_captaincy
    except Exception as e:
        print(f"Error calculating actual points: {e}")
        return predicted_team_df, 0, "Unknown", 0

def analyze_predicted_teams(team_files: List[Dict], avg_points: Dict[int, float], 
                          actual_data_base_path: str) -> pd.DataFrame:
    """
    Analyze predicted teams and calculate their actual points
    
    Args:
        team_files: List of dictionaries with team file details
        avg_points: Dictionary mapping gameweeks to average points
        actual_data_base_path: Base path to actual gameweek data
        
    Returns:
        DataFrame with analysis results
    """
    results = []
    
    for team_info in team_files:
        gw = team_info['gameweek']
        algorithm = team_info['algorithm']
        target_gw = team_info['target_gameweek']
        
        print(f"\nAnalyzing {algorithm} prediction for GW{gw} (targeting GW{target_gw})...")
        
        # Check if we have average points data for this gameweek
        if target_gw not in avg_points:
            print(f"Warning: No average points data for GW{target_gw}")
            avg_pts = None
        else:
            avg_pts = avg_points[target_gw]
            
        # Load predicted team
        try:
            predicted_team = pd.read_csv(team_info['filepath'])
            
            # Build path to actual gameweek data
            actual_data_path = os.path.join(actual_data_base_path, f"gw{target_gw}.csv")
            
            if not os.path.exists(actual_data_path):
                print(f"Error: Actual data file not found for GW{target_gw}")
                continue
                
            # Calculate actual points with captaincy simulation
            updated_team, total_points_with_captaincy, captain_name, total_points_no_captaincy = get_actual_team_points(predicted_team, actual_data_path)
            
            # Calculate average points for this team formation
            def_count = updated_team[updated_team['position_label'] == 'DEF'].shape[0]
            mid_count = updated_team[updated_team['position_label'] == 'MID'].shape[0]
            fwd_count = updated_team[updated_team['position_label'] == 'FWD'].shape[0]
            formation = f"{def_count}-{mid_count}-{fwd_count}"
            
            # Get predicted points total
            predicted_points_total = updated_team['predicted_points'].sum()
            
            # Calculate prediction accuracy metrics using points WITHOUT captaincy
            prediction_diff = -(total_points_no_captaincy - predicted_points_total)
            prediction_diff_pct = (total_points_no_captaincy / predicted_points_total - 1) * 100 if predicted_points_total > 0 else 0
            
            # Calculate prediction accuracy for this gameweek
            if total_points_no_captaincy > 0:
                prediction_accuracy = (1 - abs(prediction_diff) / total_points_no_captaincy) * 100
                # Cap at 0% to avoid negative accuracy values
                prediction_accuracy = max(0, prediction_accuracy)
            else:
                prediction_accuracy = 0
            
            # Calculate player-level prediction accuracy
            updated_team['points_diff'] = updated_team['actual_points'] - updated_team['predicted_points']
            
            avg_player_diff = updated_team['points_diff'].mean()
            std_player_diff = updated_team['points_diff'].std()
            
            # Create result entry
            result = {
                'prediction_gw': gw,
                'target_gw': target_gw,
                'algorithm': algorithm,
                'formation': formation,
                'predicted_points': predicted_points_total,
                'actual_points': total_points_with_captaincy,  # With captaincy for FPL comparison
                'actual_points_no_captaincy': total_points_no_captaincy,  # Without captaincy for prediction comparison
                'avg_fpl_points': avg_pts,
                'captain': captain_name,
                'captain_points': updated_team[updated_team['is_captain']]['actual_points'].values[0] * 2,
                'prediction_diff': prediction_diff,  # Positive = underestimated, Negative = overestimated
                'prediction_diff_pct': prediction_diff_pct,
                'prediction_accuracy': prediction_accuracy,  # Add prediction accuracy for each gameweek
                'avg_player_diff': avg_player_diff,
                'std_player_diff': std_player_diff
            }
            
            # Calculate differential from average if available
            if avg_pts is not None:
                result['points_vs_avg'] = total_points_with_captaincy - avg_pts
                result['pct_vs_avg'] = (total_points_with_captaincy / avg_pts - 1) * 100
            
            results.append(result)
            
            # Save updated team with actual points and captain info
            output_dir = os.path.dirname(team_info['filepath'])
            output_path = os.path.join(output_dir, f"predicted_team_{algorithm}_gw{gw:02d}_with_actual.csv")
            updated_team.to_csv(output_path, index=False)
            print(f"Team scored {total_points_with_captaincy:.1f} actual points in GW{target_gw} (with captaincy)")
            print(f"Team scored {total_points_no_captaincy:.1f} actual points without captaincy")
            print(f"Predicted points: {predicted_points_total:.1f}, Difference: {prediction_diff:.1f} ({prediction_diff_pct:.1f}%)")
            print(f"Prediction accuracy: {prediction_accuracy:.1f}%")
            
            if avg_pts is not None:
                print(f"Average FPL score for GW{target_gw}: {avg_pts:.1f}")
                print(f"Difference from average: {result['points_vs_avg']:.1f} points ({result['pct_vs_avg']:.1f}%)")
            
        except Exception as e:
            print(f"Error processing {algorithm} for GW{gw}: {e}")
            import traceback
            traceback.print_exc()
    
    # Convert results to DataFrame
    if not results:
        print("No results to analyze")
        return pd.DataFrame()
        
    return pd.DataFrame(results)

def create_visualizations(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create visualizations from the analysis results
    
    Args:
        results_df: DataFrame with analysis results
        output_dir: Directory to save visualizations
    """
    if results_df.empty:
        print("No data for visualizations")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set visualization style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Points vs Average by Algorithm
    plt.figure(figsize=(12, 8))
    avg_by_algo = results_df.groupby('algorithm')[['points_vs_avg', 'pct_vs_avg']].mean().reset_index()
    avg_by_algo = avg_by_algo.sort_values('points_vs_avg', ascending=False)
    
    sns.barplot(x='algorithm', y='points_vs_avg', data=avg_by_algo)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Average Points Above/Below FPL Average by Algorithm (With Captaincy)', fontsize=14)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Points vs Average', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(avg_by_algo['points_vs_avg']):
        plt.text(i, v + (0.5 if v >= 0 else -2), f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'points_vs_avg_by_algorithm.png'), dpi=150)
    plt.close()
    
    # 2. Performance heatmap across gameweeks by algorithm
    if len(results_df['target_gw'].unique()) > 1:
        # Create pivot table
        pivot = results_df.pivot_table(
            index='algorithm', 
            columns='target_gw', 
            values='points_vs_avg',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot, annot=True, cmap='RdYlGn', center=0, fmt='.1f')
        plt.title('Points vs Average by Algorithm and Gameweek (With Captaincy)', fontsize=14)
        plt.xlabel('Gameweek', fontsize=12)
        plt.ylabel('Algorithm', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'points_vs_avg_heatmap.png'), dpi=150)
        plt.close()
    
    # 3a. Cumulative difference from average over time
    if len(results_df['target_gw'].unique()) > 1:
        # Group by algorithm and gameweek, calculate mean if multiple entries
        gw_performance = results_df.groupby(['algorithm', 'target_gw'])['points_vs_avg'].mean().reset_index()
        
        # Calculate cumulative performance
        cumulative = pd.DataFrame()
        
        for algo in gw_performance['algorithm'].unique():
            algo_data = gw_performance[gw_performance['algorithm'] == algo].sort_values('target_gw')
            algo_data['cumulative_vs_avg'] = algo_data['points_vs_avg'].cumsum()
            cumulative = pd.concat([cumulative, algo_data])
        
        plt.figure(figsize=(14, 8))
        sns.lineplot(data=cumulative, x='target_gw', y='cumulative_vs_avg', hue='algorithm', marker='o', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.title('Cumulative Points vs Average by Algorithm Over Time (With Captaincy)', fontsize=14)
        plt.xlabel('Gameweek', fontsize=12)
        plt.ylabel('Cumulative Points vs Average', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Algorithm')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cumulative_performance_diff.png'), dpi=150)
        plt.close()
    
    # 3b. NEW: Actual cumulative points over time (including average)
    if len(results_df['target_gw'].unique()) > 1:
        # Group by algorithm and gameweek for actual points
        actual_points_by_gw = results_df.groupby(['algorithm', 'target_gw'])['actual_points'].mean().reset_index()
        
        # Create a dataframe for average FPL points
        avg_points_by_gw = results_df[['target_gw', 'avg_fpl_points']].drop_duplicates().reset_index(drop=True)
        avg_points_by_gw['algorithm'] = 'Average FPL'
        avg_points_by_gw = avg_points_by_gw.rename(columns={'avg_fpl_points': 'actual_points'})
        
        # Combine algorithm data with average data
        all_points_data = pd.concat([actual_points_by_gw, avg_points_by_gw])
        
        # Calculate cumulative points
        cumulative_points = pd.DataFrame()
        
        for algo in all_points_data['algorithm'].unique():
            algo_data = all_points_data[all_points_data['algorithm'] == algo].sort_values('target_gw')
            algo_data['cumulative_points'] = algo_data['actual_points'].cumsum()
            cumulative_points = pd.concat([cumulative_points, algo_data])
        
        # Plot cumulative points
        plt.figure(figsize=(14, 8))
        
        # Use a different line style for Average FPL
        for algo in cumulative_points['algorithm'].unique():
            algo_data = cumulative_points[cumulative_points['algorithm'] == algo]
            
            if algo == 'Average FPL':
                # Make average line more distinct
                plt.plot(algo_data['target_gw'], algo_data['cumulative_points'], 
                         'r--', linewidth=3, marker='*', markersize=10, label=algo)
            else:
                plt.plot(algo_data['target_gw'], algo_data['cumulative_points'], 
                         marker='o', linewidth=2, label=algo)
        
        plt.title('Actual Cumulative Points by Algorithm Over Time (With Captaincy)', fontsize=14)
        plt.xlabel('Gameweek', fontsize=12)
        plt.ylabel('Cumulative Points', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Algorithm')
        
        # Set integer x-ticks
        if len(results_df['target_gw'].unique()) <= 20:  # Only if not too many gameweeks
            plt.xticks(sorted(results_df['target_gw'].unique()))
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cumulative_actual_points.png'), dpi=150)
        plt.close()
    
    # 4. Actual points by algorithm boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='algorithm', y='actual_points', data=results_df)
    plt.title('Distribution of Actual Points by Algorithm (With Captaincy)', fontsize=14)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Actual Points', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_points_boxplot.png'), dpi=150)
    plt.close()
    
    # 5. Algorithm predicted vs actual points scatter plot (using points without captaincy for fair comparison)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=results_df, x='predicted_points', y='actual_points_no_captaincy', hue='algorithm', s=100, alpha=0.7)
    
    # Add diagonal line for perfect prediction
    max_val = max(results_df['predicted_points'].max(), results_df['actual_points_no_captaincy'].max())
    min_val = min(results_df['predicted_points'].min(), results_df['actual_points_no_captaincy'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.title('Predicted vs Actual Points by Algorithm (Without Captaincy)', fontsize=14)
    plt.xlabel('Predicted Points', fontsize=12)
    plt.ylabel('Actual Points (Without Captaincy)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'), dpi=150)
    plt.close()
    
    # 6. Captain points contribution by algorithm
    plt.figure(figsize=(12, 8))
    captain_contribution = results_df.copy()
    captain_contribution['captain_contribution_pct'] = (captain_contribution['captain_points'] / captain_contribution['actual_points']) * 100
    
    sns.barplot(x='algorithm', y='captain_contribution_pct', data=captain_contribution)
    plt.title('Captain Points Contribution by Algorithm (%)', fontsize=14)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Captain Points Contribution (%)', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(captain_contribution.groupby('algorithm')['captain_contribution_pct'].mean()):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'captain_contribution.png'), dpi=150)
    plt.close()
    
    # 7. NEW: Prediction accuracy by algorithm (using individual gameweek accuracies)
    plt.figure(figsize=(12, 8))
    # Group by algorithm and calculate mean of individual gameweek accuracies
    accuracy_by_algo = results_df.groupby('algorithm')['prediction_accuracy'].mean().reset_index()
    accuracy_by_algo = accuracy_by_algo.sort_values('prediction_accuracy', ascending=False)
    
    # Create accuracy bar chart
    sns.barplot(x='algorithm', y='prediction_accuracy', data=accuracy_by_algo)
    plt.title('Average Prediction Accuracy by Algorithm (Higher is Better)', fontsize=14)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Average Prediction Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(accuracy_by_algo['prediction_accuracy']):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_accuracy.png'), dpi=150)
    plt.close()
    
    # 8. Prediction accuracy distribution by algorithm
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='algorithm', y='prediction_accuracy', data=results_df)
    plt.title('Distribution of Prediction Accuracy by Algorithm', fontsize=14)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Prediction Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_accuracy_boxplot.png'), dpi=150)
    plt.close()
    
    # 9. Prediction difference distribution by algorithm
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='algorithm', y='prediction_diff', data=results_df)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('Distribution of Prediction Difference by Algorithm (Without Captaincy)', fontsize=14)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Prediction Difference (Actual - Predicted)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_diff_boxplot.png'), dpi=150)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def generate_summary_report(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate a summary report of the analysis
    
    Args:
        results_df: DataFrame with analysis results
        output_dir: Directory to save the report
    """
    if results_df.empty:
        print("No data for summary report")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a summary table by algorithm
    summary = results_df.groupby('algorithm').agg({
        'actual_points': ['mean', 'min', 'max', 'std'],
        'points_vs_avg': ['mean', 'min', 'max', 'std'] if 'points_vs_avg' in results_df.columns else ['size'],
        'pct_vs_avg': ['mean', 'min', 'max', 'std'] if 'pct_vs_avg' in results_df.columns else ['size'],
        'target_gw': 'count',
        'captain_points': ['mean', 'sum'],  # Captain points statistics
        'prediction_diff': ['mean', 'std', 'min', 'max'],  # Prediction accuracy
        'prediction_diff_pct': ['mean', 'std'],
        'avg_player_diff': ['mean'],
        'std_player_diff': ['mean']
    }).reset_index()
    
    # Flatten the multi-index columns
    summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary.columns.values]
    
    # Rename columns for clarity
    rename_dict = {
        'actual_points_mean': 'avg_points',
        'target_gw_count': 'gameweeks_analyzed',
        'captain_points_mean': 'avg_captain_points',
        'captain_points_sum': 'total_captain_points',
        'prediction_diff_mean': 'avg_prediction_diff',
        'prediction_diff_std': 'std_prediction_diff',
        'prediction_diff_min': 'min_prediction_diff',
        'prediction_diff_max': 'max_prediction_diff',
        'prediction_diff_pct_mean': 'avg_prediction_diff_pct',
        'prediction_diff_pct_std': 'std_prediction_diff_pct',
        'avg_player_diff_mean': 'avg_player_prediction_diff',
        'std_player_diff_mean': 'avg_std_player_prediction_diff'
    }
    
    if 'points_vs_avg_mean' in summary.columns:
        rename_dict['points_vs_avg_mean'] = 'avg_vs_avg'
        rename_dict['pct_vs_avg_mean'] = 'avg_pct_vs_avg'
    
    summary = summary.rename(columns=rename_dict)
    
    # Calculate captain contribution percentage
    summary['captain_contribution_pct'] = (summary['avg_captain_points'] / summary['avg_points']) * 100
    
    # Calculate prediction accuracy percentage
    summary['prediction_accuracy'] = (1 - abs(summary['avg_prediction_diff']) / summary['avg_points']) * 100
    
    # Sort by average points (descending)
    summary = summary.sort_values('avg_points', ascending=False)
    
    # Calculate win rate (percentage of gameweeks where algorithm beat the average)
    if 'points_vs_avg' in results_df.columns:
        win_rates = []
        for algo in summary['algorithm']:
            algo_results = results_df[results_df['algorithm'] == algo]
            wins = (algo_results['points_vs_avg'] > 0).sum()
            win_rate = wins / len(algo_results) * 100
            win_rates.append(win_rate)
        
        summary['win_rate_pct'] = win_rates
    
    # Save summary to CSV
    summary_path = os.path.join(output_dir, 'algorithm_performance_summary.csv')
    summary.to_csv(summary_path, index=False)
    
    # Also save the detailed results
    detailed_path = os.path.join(output_dir, 'detailed_gameweek_results.csv')
    results_df.to_csv(detailed_path, index=False)
    
    print(f"Summary report saved to {output_dir}")
    
    # Print text summary to console
    print("\n===== ALGORITHM PERFORMANCE SUMMARY (WITH CAPTAINCY) =====")
    print(f"Data from {summary['gameweeks_analyzed'].sum()} predictions across {len(summary)} algorithms")
    print("\nRanked by average actual points:")
    
    # Format for display
    for i, (_, row) in enumerate(summary.iterrows()):
        output_str = f"{i+1}. {row['algorithm']}: {row['avg_points']:.1f} pts"
        
        if 'avg_vs_avg' in row:
            output_str += f" ({row['avg_vs_avg']:.1f} vs avg, {row['avg_pct_vs_avg']:.1f}%)"
        
        if 'win_rate_pct' in row:
            output_str += f", Win rate: {row['win_rate_pct']:.1f}%"
        
        output_str += f", Captain contribution: {row['captain_contribution_pct']:.1f}%"
        
        # Add prediction accuracy stats
        output_str += f"\n   Prediction diff: {row['avg_prediction_diff']:.1f} pts (σ={row['std_prediction_diff']:.1f}), Accuracy: {row['prediction_accuracy']:.1f}%"
        output_str += f"\n   Player-level avg diff: {row['avg_player_prediction_diff']:.2f} pts, avg std: {row['avg_std_player_prediction_diff']:.2f}"
            
        print(output_str)

def main():
    parser = argparse.ArgumentParser(description='Analyze actual points earned by predicted FPL teams (with captaincy)')
    
    parser.add_argument('--predictions-path', type=str, default='predictions',
                      help='Base path to predictions directory')
    
    parser.add_argument('--actual-data-path', type=str, 
                      default=os.path.join('Up-to-data', 'Fantasy-Premier-League', 'data', '2024-25', 'gws'),
                      help='Base path to actual gameweek data')
    
    parser.add_argument('--avg-points-path', type=str, default=os.path.join('data', 'avg_points.csv'),
                      help='Path to average points per gameweek data')
    
    parser.add_argument('--output-dir', type=str, default='analysis_results_with_captaincy',
                      help='Directory to save analysis results')
    
    parser.add_argument('--algorithm', type=str, default=None,
                      help='Analyze only a specific algorithm')
    
    args = parser.parse_args()
    
    # Find all predicted team files
    print(f"Searching for predicted team files in {args.predictions_path}...")
    team_files = find_predicted_teams(args.predictions_path)
    
    if not team_files:
        print("No predicted team files found. Please check the predictions path.")
        return
    
    # Filter by algorithm if specified
    if args.algorithm:
        team_files = [tf for tf in team_files if tf['algorithm'] == args.algorithm]
        if not team_files:
            print(f"No team files found for algorithm: {args.algorithm}")
            return
    
    print(f"Found {len(team_files)} predicted team files")
    
    # Load average points data
    print(f"Loading average points data from {args.avg_points_path}...")
    avg_points = load_average_points(args.avg_points_path)
    
    if not avg_points:
        print("Warning: No average points data loaded. Continuing without comparison to average.")
    
    # Analyze predicted teams
    print("\nAnalyzing predicted teams with captaincy simulation...")
    results_df = analyze_predicted_teams(team_files, avg_points, args.actual_data_path)
    
    if results_df.empty:
        print("No results to report")
        return
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(results_df, args.output_dir)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results_df, os.path.join(args.output_dir, 'figures'))
    
    print("\nAnalysis with captaincy simulation complete!")

if __name__ == "__main__":
    main()
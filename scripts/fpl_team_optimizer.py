import os
import pandas as pd
import numpy as np
import requests
import random
import json
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Set, Optional, Union
from collections import defaultdict
import argparse
import concurrent.futures
import time

# Function to load prediction data from a specific algorithm's CSV file
def load_predictions(gw: int, algorithm: str) -> pd.DataFrame:
    """
    Load predictions for a specific algorithm and gameweek
    
    Args:
        gw: Gameweek number
        algorithm: Algorithm name (DeepLearning, GBReg, LSTM, RFReg, Ridge, XGBoost)
        
    Returns:
        DataFrame with predictions
    """
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the path to the predictions file
    filepath = os.path.join(script_dir, "..", "predictions", f"gw{gw:02d}", algorithm, "predictions", f"predictions_{algorithm}_regression.csv")
    
    try:
        df = pd.read_csv(filepath)
        # Ensure the dataframe has the expected columns
        if not all(col in df.columns for col in ['name', 'position_label', 'predicted_points']):
            raise ValueError(f"CSV file {filepath} is missing required columns")
        return df
    except FileNotFoundError:
        print(f"Prediction file not found: {filepath}")
        return pd.DataFrame(columns=['name', 'position_label', 'predicted_points'])


# Function to find available gameweeks with prediction data
def find_available_gameweeks(algorithms: List[str]) -> List[int]:
    """
    Find gameweeks for which prediction data exists
    
    Args:
        algorithms: List of algorithms to check
        
    Returns:
        List of available gameweeks
    """
    available_gws = set()
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the base path to the predictions directory
    base_path = os.path.join(script_dir, "..", "predictions")
    
    # Look for gameweeks in the predictions directory
    if os.path.exists(base_path):
        # List all subdirectories that match gw pattern
        gw_dirs = [d for d in os.listdir(base_path) if d.startswith('gw')]
        
        for gw_dir in gw_dirs:
            # Extract the gameweek number
            match = re.search(r'gw(\d+)', gw_dir)
            if match:
                gw_num = int(match.group(1))
                
                # Check if prediction files exist for any of the specified algorithms
                for algorithm in algorithms:
                    pred_path = os.path.join(base_path, gw_dir, algorithm, "predictions", f"predictions_{algorithm}_regression.csv")
                    if os.path.exists(pred_path):
                        available_gws.add(gw_num)
                        break  # Found at least one algorithm for this gameweek
    
    return sorted(list(available_gws))

# Function to load actual gameweek data for ground truth comparison
def load_actual_gameweek_data(gw: int) -> pd.DataFrame:
    """
    Load actual gameweek data for creating the ground truth optimal team
    
    Args:
        gw: Gameweek number to load
        
    Returns:
        DataFrame with actual gameweek data
    """
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the path to the actual gameweek data
    filepath = os.path.join(script_dir, "..", "Up-to-data", "Fantasy-Premier-League", "data", "2024-25", "gws", f"gw{gw}.csv")
    
    try:
        df = pd.read_csv(filepath)
        # Check if necessary columns exist
        required_cols = ['name', 'position', 'total_points']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Warning: Missing columns in actual gameweek data: {missing}")
            
            # Handle common alternatives
            if 'total_points' not in df.columns and 'points' in df.columns:
                df['total_points'] = df['points']
            
        # Standardize position column to match format used in predictions
        if 'position' in df.columns:
            # Rename position column to position_label to match prediction data
            df['position_label'] = df['position']
            
            # Ensure GKP is renamed to GK for consistency
            df['position_label'] = df['position_label'].replace('GKP', 'GK')
            
        return df
    except FileNotFoundError:
        print(f"Actual gameweek data file not found: {filepath}")
        return pd.DataFrame(columns=['name', 'position_label', 'total_points'])

# Function to fetch player data from FPL API
def fetch_fpl_data() -> Dict:
    """
    Fetch player data from FPL API
    
    Returns:
        Dictionary mapping player names to their chance of playing next round
    """
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()
        
        # Create mapping from player name to chance_of_playing_next_round
        player_chances = {}
        for player in data['elements']:
            name = f"{player['first_name']} {player['second_name']}"
            chance = player['chance_of_playing_next_round']
            # If chance is None, assume 100%
            player_chances[name] = 100 if chance is None else chance
            
        return player_chances
    except requests.RequestException as e:
        print(f"Error fetching FPL data: {e}")
        return {}

# Function to merge predictions with FPL data
def merge_predictions_with_fpl(predictions: pd.DataFrame, fpl_data: Dict) -> pd.DataFrame:
    """
    Merge prediction data with FPL player fitness data
    
    Args:
        predictions: DataFrame with player predictions
        fpl_data: Dictionary mapping player names to chance of playing
        
    Returns:
        DataFrame with predictions and fitness data
    """
    # Add chance_of_playing_next_round column
    merged_df = predictions.copy()
    merged_df['chance_of_playing_next_round'] = merged_df['name'].map(fpl_data).fillna(100)
    
    # Adjust predicted points based on chance of playing
    # Scale points by the chance of playing percentage
    merged_df['adjusted_points'] = merged_df['predicted_points'] * (merged_df['chance_of_playing_next_round'] / 100)
    
    return merged_df

# Custom class to represent a team
class Team:
    def __init__(self, players: List[Dict], use_adjusted_points: bool = True):
        self.players = players
        
        # Determine which points field to use for fitness
        self.points_field = 'adjusted_points' if use_adjusted_points else 'predicted_points'
        
        # For actual teams, use total_points
        if 'total_points' in players[0]:
            self.points_field = 'total_points'
            
        self.total_points = sum(player[self.points_field] for player in players)
        
        # Count positions
        self.position_counts = {
            'GK': sum(1 for p in players if p['position_label'] == 'GK'),
            'DEF': sum(1 for p in players if p['position_label'] == 'DEF'),
            'MID': sum(1 for p in players if p['position_label'] == 'MID'),
            'FWD': sum(1 for p in players if p['position_label'] == 'FWD')
        }
        
        # Check if team is valid
        self.is_valid = (
            len(players) == 11 and
            self.position_counts['GK'] == 1 and
            3 <= self.position_counts['DEF'] <= 5 and
            2 <= self.position_counts['MID'] <= 5 and
            1 <= self.position_counts['FWD'] <= 3
        )
        
        # Calculate constraint violation penalty
        self.constraint_violation = 0
        if len(players) != 11:
            self.constraint_violation += abs(len(players) - 11) * 10
        if self.position_counts['GK'] != 1:
            self.constraint_violation += abs(self.position_counts['GK'] - 1) * 10
        if self.position_counts['DEF'] < 3:
            self.constraint_violation += (3 - self.position_counts['DEF']) * 10
        elif self.position_counts['DEF'] > 5:
            self.constraint_violation += (self.position_counts['DEF'] - 5) * 10
        if self.position_counts['MID'] < 2:
            self.constraint_violation += (2 - self.position_counts['MID']) * 10
        elif self.position_counts['MID'] > 5:
            self.constraint_violation += (self.position_counts['MID'] - 5) * 10
        if self.position_counts['FWD'] < 1:
            self.constraint_violation += (1 - self.position_counts['FWD']) * 10
        elif self.position_counts['FWD'] > 3:
            self.constraint_violation += (self.position_counts['FWD'] - 3) * 10
            
    def get_player_names(self) -> Set[str]:
        """Return a set of player names in the team"""
        return {player['name'] for player in self.players}
    
    def get_formation(self) -> str:
        """Return the team formation as a string (e.g. '4-4-2')"""
        return f"{self.position_counts['DEF']}-{self.position_counts['MID']}-{self.position_counts['FWD']}"

# Genetic Algorithm Implementation
class GeneticAlgorithm:
    def __init__(self, player_pool: pd.DataFrame, pop_size: int = 100, max_generations: int = 50, 
                 stagnation_limit: int = 10, use_adjusted_points: bool = True, verbose: bool = True):
        self.player_pool = player_pool
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.stagnation_limit = stagnation_limit
        self.use_adjusted_points = use_adjusted_points
        self.verbose = verbose
        
        # Determine points field
        self.points_field = 'adjusted_points' if use_adjusted_points else 'predicted_points'
        if 'total_points' in player_pool.columns and not use_adjusted_points:
            self.points_field = 'total_points'
        
        # Group players by position for initialization
        self.players_by_position = {
            'GK': player_pool[player_pool['position_label'] == 'GK'].index.tolist(),
            'DEF': player_pool[player_pool['position_label'] == 'DEF'].index.tolist(),
            'MID': player_pool[player_pool['position_label'] == 'MID'].index.tolist(),
            'FWD': player_pool[player_pool['position_label'] == 'FWD'].index.tolist()
        }
    
    def initialize_population(self) -> List[List[int]]:
        """Create initial population with pseudo-random initialization"""
        population = []
        for _ in range(self.pop_size):
            # Generate a valid team
            individual = self._generate_valid_individual()
            population.append(individual)
        return population
    
    def _generate_valid_individual(self) -> List[int]:
        """Generate a valid team satisfying all constraints"""
        team = []
        
        # Add 1 goalkeeper
        if self.players_by_position['GK']:
            team.extend(random.sample(self.players_by_position['GK'], min(1, len(self.players_by_position['GK']))))
        
        # Add 3-5 defenders
        num_def = random.randint(3, 5)
        if len(self.players_by_position['DEF']) >= num_def:
            team.extend(random.sample(self.players_by_position['DEF'], num_def))
        else:
            team.extend(self.players_by_position['DEF'])
        
        # Add 2-5 midfielders (ensuring total doesn't exceed 10 to leave room for at least 1 forward)
        max_mid = min(5, 10 - len(team) - 1)
        num_mid = random.randint(2, max_mid)
        if len(self.players_by_position['MID']) >= num_mid:
            team.extend(random.sample(self.players_by_position['MID'], num_mid))
        else:
            team.extend(self.players_by_position['MID'])
        
        # Add 1-3 forwards (fill remaining slots)
        remaining = 11 - len(team)
        num_fwd = min(3, remaining)
        num_fwd = max(1, num_fwd)  # Ensure at least 1 forward
        if len(self.players_by_position['FWD']) >= num_fwd:
            team.extend(random.sample(self.players_by_position['FWD'], num_fwd))
        else:
            team.extend(self.players_by_position['FWD'])
        
        return team
    
    def evaluate_fitness(self, individual: List[int]) -> float:
        """Evaluate fitness of an individual"""
        selected_players = [self.player_pool.iloc[i].to_dict() for i in individual]
        team = Team(selected_players, use_adjusted_points=self.use_adjusted_points)
        
        if team.is_valid:
            return team.total_points
        else:
            # Penalize invalid teams
            return team.total_points - team.constraint_violation * 10
    
    def stochastic_ranking(self, population: List[List[int]], pf: float = 0.45) -> List[List[int]]:
        """Stochastic ranking selection as described by Runarsson and Yao"""
        n = len(population)
        ranking = list(range(n))
        
        # Calculate fitness and constraint violations for all individuals
        fitness_values = [self.evaluate_fitness(ind) for ind in population]
        violations = [self._calculate_constraint_violation(ind) for ind in population]
        
        # Stochastic bubble sort
        for _ in range(n):
            swapped = False
            for j in range(n - 1):
                u = random.random()
                
                # Sort based on constraint violation with probability 1-pf
                # Or sort based on fitness if both feasible
                if violations[ranking[j]] == 0 and violations[ranking[j+1]] == 0:
                    # Both feasible, compare fitness
                    if fitness_values[ranking[j]] < fitness_values[ranking[j+1]]:
                        ranking[j], ranking[j+1] = ranking[j+1], ranking[j]
                        swapped = True
                elif u < pf:
                    # Compare using fitness with probability pf
                    if fitness_values[ranking[j]] < fitness_values[ranking[j+1]]:
                        ranking[j], ranking[j+1] = ranking[j+1], ranking[j]
                        swapped = True
                else:
                    # Compare using constraint violation
                    if violations[ranking[j]] > violations[ranking[j+1]]:
                        ranking[j], ranking[j+1] = ranking[j+1], ranking[j]
                        swapped = True
            
            if not swapped:
                break
        
        # Return sorted population
        return [population[ranking[i]] for i in range(n)]
    
    def _calculate_constraint_violation(self, individual: List[int]) -> float:
        """Calculate constraint violation for an individual"""
        selected_players = [self.player_pool.iloc[i].to_dict() for i in individual]
        team = Team(selected_players, use_adjusted_points=self.use_adjusted_points)
        return team.constraint_violation
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Perform crossover between two parents"""
        # Determine crossover point (ensuring it's at a position boundary)
        positions = []
        
        # Map each index to its position
        for i in range(len(parent1)):
            player_idx = parent1[i]
            position = self.player_pool.iloc[player_idx]['position_label']
            positions.append(position)
        
        # Find position boundaries
        boundaries = []
        current_pos = positions[0]
        
        for i in range(1, len(positions)):
            if positions[i] != current_pos:
                boundaries.append(i)
                current_pos = positions[i]
        
        if not boundaries:
            # No boundaries found, use random point
            crossover_point = random.randint(1, len(parent1) - 1)
        else:
            # Choose a position boundary
            crossover_point = random.choice(boundaries)
        
        # Create offspring
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        # Repair offspring to ensure they're valid
        offspring1 = self._repair_individual(offspring1)
        offspring2 = self._repair_individual(offspring2)
        
        return offspring1, offspring2
    
    def mutate(self, individual: List[int], mutation_rate: float = 0.1) -> List[int]:
        """Perform mutation on an individual"""
        mutated = individual.copy()
        
        # Group the current team by position
        team_by_pos = {
            'GK': [],
            'DEF': [],
            'MID': [],
            'FWD': []
        }
        
        for idx in mutated:
            pos = self.player_pool.iloc[idx]['position_label']
            team_by_pos[pos].append(idx)
        
        # Randomly mutate players by position
        for pos in team_by_pos:
            for i, player_idx in enumerate(team_by_pos[pos]):
                if random.random() < mutation_rate:
                    # Get available players of the same position not in the team
                    available = [p for p in self.players_by_position[pos] if p not in mutated]
                    
                    if available:
                        # Replace with random available player
                        replacement = random.choice(available)
                        
                        # Find the index in the team
                        old_index = mutated.index(player_idx)
                        mutated[old_index] = replacement
        
        # Repair to ensure team is valid
        return self._repair_individual(mutated)
    
    def _repair_individual(self, individual: List[int]) -> List[int]:
        """Repair an individual to ensure it meets all constraints"""
        # Remove duplicates
        individual = list(dict.fromkeys(individual))
        
        # Group by position
        team_by_pos = {
            'GK': [],
            'DEF': [],
            'MID': [],
            'FWD': []
        }
        
        for idx in individual:
            pos = self.player_pool.iloc[idx]['position_label']
            team_by_pos[pos].append(idx)
        
        # Fix goalkeeper count (exactly 1)
        while len(team_by_pos['GK']) > 1:
            team_by_pos['GK'].pop()
        
        while len(team_by_pos['GK']) < 1:
            available = [p for p in self.players_by_position['GK'] if p not in individual]
            if available:
                team_by_pos['GK'].append(random.choice(available))
            else:
                break  # Not enough goalkeepers available
        
        # Fix defender count (3-5)
        while len(team_by_pos['DEF']) > 5:
            team_by_pos['DEF'].pop()
        
        while len(team_by_pos['DEF']) < 3:
            available = [p for p in self.players_by_position['DEF'] if p not in individual]
            if available:
                team_by_pos['DEF'].append(random.choice(available))
            else:
                break  # Not enough defenders available
        
        # Fix midfielder count (2-5)
        while len(team_by_pos['MID']) > 5:
            team_by_pos['MID'].pop()
        
        while len(team_by_pos['MID']) < 2:
            available = [p for p in self.players_by_position['MID'] if p not in individual]
            if available:
                team_by_pos['MID'].append(random.choice(available))
            else:
                break  # Not enough midfielders available
        
        # Fix forward count (1-3)
        while len(team_by_pos['FWD']) > 3:
            team_by_pos['FWD'].pop()
        
        while len(team_by_pos['FWD']) < 1:
            available = [p for p in self.players_by_position['FWD'] if p not in individual]
            if available:
                team_by_pos['FWD'].append(random.choice(available))
            else:
                break  # Not enough forwards available
        
        # Combine positions
        repaired = []
        for pos in team_by_pos:
            repaired.extend(team_by_pos[pos])
        
        # Ensure exactly 11 players
        while len(repaired) > 11:
            # Remove excess players prioritizing positions with more flexibility
            for pos in ['MID', 'DEF', 'FWD', 'GK']:
                if team_by_pos[pos] and (
                    (pos == 'MID' and len(team_by_pos[pos]) > 2) or
                    (pos == 'DEF' and len(team_by_pos[pos]) > 3) or
                    (pos == 'FWD' and len(team_by_pos[pos]) > 1) or
                    (pos == 'GK' and len(team_by_pos[pos]) > 1)
                ):
                    removed = team_by_pos[pos].pop()
                    repaired.remove(removed)
                    if len(repaired) == 11:
                        break
            
            # If still over 11, remove randomly
            if len(repaired) > 11:
                repaired.pop(random.randrange(len(repaired)))
        
        while len(repaired) < 11:
            # Add players prioritizing positions that need more
            positions_to_try = []
            
            if len(team_by_pos['GK']) < 1:
                positions_to_try.append('GK')
            if len(team_by_pos['DEF']) < 3:
                positions_to_try.append('DEF')
            if len(team_by_pos['MID']) < 2:
                positions_to_try.append('MID')
            if len(team_by_pos['FWD']) < 1:
                positions_to_try.append('FWD')
            
            # If no required positions, add to positions with room
            if not positions_to_try:
                if len(team_by_pos['DEF']) < 5:
                    positions_to_try.append('DEF')
                if len(team_by_pos['MID']) < 5:
                    positions_to_try.append('MID')
                if len(team_by_pos['FWD']) < 3:
                    positions_to_try.append('FWD')
            
            # If still no positions, team can't be repaired
            if not positions_to_try:
                break
            
            # Try to add a player from a random position
            pos = random.choice(positions_to_try)
            available = [p for p in self.players_by_position[pos] if p not in repaired]
            
            if available:
                new_player = random.choice(available)
                repaired.append(new_player)
                team_by_pos[pos].append(new_player)
            else:
                # No available players for this position, try another
                positions_to_try.remove(pos)
                if not positions_to_try:
                    break
        
        return repaired
    
    def heuristic_improvement(self, individual: List[int]) -> List[int]:
        """Apply heuristic improvement to an individual"""
        improved = individual.copy()
        
        # Group by position
        team_by_pos = {
            'GK': [],
            'DEF': [],
            'MID': [],
            'FWD': []
        }
        
        for idx in improved:
            pos = self.player_pool.iloc[idx]['position_label']
            team_by_pos[pos].append(idx)
        
        # For each position, try to replace the lowest scoring player
        for pos in team_by_pos:
            if team_by_pos[pos]:
                # Find the player with lowest points
                worst_idx = min(team_by_pos[pos], key=lambda idx: self.player_pool.iloc[idx][self.points_field])
                worst_points = self.player_pool.iloc[worst_idx][self.points_field]
                
                # Find better players not in the team
                better_players = [
                    p for p in self.players_by_position[pos] 
                    if p not in improved and 
                    self.player_pool.iloc[p][self.points_field] > worst_points
                ]
                
                if better_players:
                    # Replace with the best available player
                    best_replacement = max(better_players, key=lambda idx: self.player_pool.iloc[idx][self.points_field])
                    
                    # Replace in the team
                    team_by_pos[pos].remove(worst_idx)
                    team_by_pos[pos].append(best_replacement)
                    
                    # Update the individual
                    improved[improved.index(worst_idx)] = best_replacement
        
        # Ensure team is still valid
        return self._repair_individual(improved)
    
    def run(self) -> List[int]:
        """Run the genetic algorithm and return the best individual"""
        # Initialize population
        population = self.initialize_population()
        
        # Track best individual
        best_individual = None
        best_fitness = float('-inf')
        
        # Counter for stagnation (generations without improvement)
        stagnation_counter = 0
        
        # Run until stagnation or max generations
        generation = 0
        while generation < self.max_generations:
            # Evaluate fitness for each individual
            fitness_values = [self.evaluate_fitness(ind) for ind in population]
            
            # Find best individual in current generation
            current_best_idx = fitness_values.index(max(fitness_values))
            current_best = population[current_best_idx]
            current_best_fitness = fitness_values[current_best_idx]
            
            # Check for improvement
            if current_best_fitness > best_fitness:
                best_individual = current_best.copy()
                best_fitness = current_best_fitness
                stagnation_counter = 0  # Reset stagnation counter
                if self.verbose:
                    print(f"Generation {generation}: New best fitness = {best_fitness}")
            else:
                stagnation_counter += 1
                if self.verbose and generation % 10 == 0:
                    print(f"Generation {generation}: Best fitness = {best_fitness} (No improvement for {stagnation_counter} generations)")
            
            # Check stagnation termination condition
            if stagnation_counter >= self.stagnation_limit:
                if self.verbose:
                    print(f"Terminating: No improvement for {stagnation_counter} generations")
                break
            
            # Stochastic ranking selection
            ranked_population = self.stochastic_ranking(population)
            
            # Select top individuals for next generation
            selected = ranked_population[:self.pop_size // 2]
            
            # Create new population
            new_population = selected.copy()
            
            # Add offspring through crossover and mutation
            while len(new_population) < self.pop_size:
                # Select parents
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                
                # Crossover
                offspring1, offspring2 = self.crossover(parent1, parent2)
                
                # Mutation
                if random.random() < 0.2:  # Mutation rate
                    offspring1 = self.mutate(offspring1)
                if random.random() < 0.2:
                    offspring2 = self.mutate(offspring2)
                
                # Add to new population
                new_population.append(offspring1)
                if len(new_population) < self.pop_size:
                    new_population.append(offspring2)
            
            # Update population
            population = new_population
            
            # Increment generation counter
            generation += 1
        
        # Apply heuristic improvement to best individual
        improved_best = self.heuristic_improvement(best_individual)
        improved_fitness = self.evaluate_fitness(improved_best)
        
        if improved_fitness > best_fitness:
            best_individual = improved_best
            best_fitness = improved_fitness
        
        if self.verbose:
            print(f"Final best fitness: {best_fitness}")
        
        return best_individual

# Function to build optimal team from actual gameweek data with improved error handling
def build_optimal_actual_team(df: pd.DataFrame, pop_size: int = 150, max_generations: int = 200) -> pd.DataFrame:
    """
    Build the optimal team based on actual points from a gameweek
    
    Args:
        df: DataFrame with actual gameweek data
        pop_size: Population size for genetic algorithm
        max_generations: Maximum generations to run
        
    Returns:
        DataFrame containing the optimal actual team
    """
    print(f"Building optimal actual team with data shape: {df.shape}")
    print(f"Columns available: {df.columns.tolist()}")
    
    if df.empty:
        print("ERROR: Empty dataframe provided to build_optimal_actual_team")
        return pd.DataFrame()
    
    # Check if required columns exist
    required_cols = ['name', 'total_points']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return pd.DataFrame()
    
    # Ensure df has position_label column
    if 'position' in df.columns and 'position_label' not in df.columns:
        df['position_label'] = df['position']
        print("Added position_label column based on position column")
    
    if 'position_label' not in df.columns:
        print("ERROR: No position or position_label column found")
        return pd.DataFrame()
    
    # Ensure the total_points column is numeric
    try:
        df['total_points'] = pd.to_numeric(df['total_points'], errors='coerce')
        if df['total_points'].isna().any():
            print(f"WARNING: {df['total_points'].isna().sum()} NaN values in total_points, filling with zeros")
            df['total_points'] = df['total_points'].fillna(0)
    except Exception as e:
        print(f"ERROR converting total_points to numeric: {e}")
        return pd.DataFrame()
    
    # Run genetic algorithm to find optimal team
    try:
        ga = GeneticAlgorithm(
            player_pool=df, 
            pop_size=pop_size, 
            max_generations=max_generations,
            stagnation_limit=50,
            use_adjusted_points=False,  # Don't use adjusted points for actual data
            verbose=False              # Less output for ground truth
        )
        
        best_indices = ga.run()
        best_team = df.iloc[best_indices].copy()
        
        # Sort by position
        best_team = best_team.sort_values(['position_label', 'total_points'], ascending=[True, False])
        
        print(f"Optimal team created with {len(best_team)} players")
        return best_team
    except Exception as e:
        print(f"ERROR in genetic algorithm: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# Function to display the best team
def display_team(team_df: pd.DataFrame, team_type: str, include_points: bool = True) -> None:
    """
    Display formatted team information
    
    Args:
        team_df: DataFrame containing team players
        team_type: String describing the team (e.g., 'Predicted', 'Actual')
        include_points: Whether to display point information
    """
    # Count positions
    position_counts = team_df['position_label'].value_counts().to_dict()
    
    # Calculate total points
    if 'adjusted_points' in team_df.columns:
        total_adjusted = team_df['adjusted_points'].sum()
        total_predicted = team_df['predicted_points'].sum()
        points_str = f"Total predicted points: {total_predicted:.2f}\n"
        points_str += f"Total adjusted points (considering fitness): {total_adjusted:.2f}"
    elif 'total_points' in team_df.columns:
        total_points = team_df['total_points'].sum()
        points_str = f"Total actual points: {total_points:.2f}"
    else:
        points_str = "No point information available"
    
    # Print results
    print(f"\n===== {team_type} Team =====")
    print(f"Formation: {position_counts.get('DEF', 0)}-{position_counts.get('MID', 0)}-{position_counts.get('FWD', 0)}")
    if include_points:
        print(points_str)
    print()
    
    # Print players by position
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        print(f"{pos}:")
        pos_players = team_df[team_df['position_label'] == pos]
        
        for _, player in pos_players.iterrows():
            name = player['name']
            
            # Determine points to display based on available columns
            if 'adjusted_points' in team_df.columns:
                points = f"{player['predicted_points']:.2f} points"
                if player['chance_of_playing_next_round'] < 100:
                    points += f" (Fitness: {player['chance_of_playing_next_round']}%)"
            elif 'total_points' in team_df.columns:
                points = f"{player['total_points']} points"
            else:
                points = ""
                
            if include_points:
                print(f"  {name}: {points}")
            else:
                print(f"  {name}")

# Function to compare predicted team with actual team
def compare_teams(predicted_team: pd.DataFrame, actual_team: pd.DataFrame) -> Dict:
    """
    Compare a predicted team with the actual optimal team
    
    Args:
        predicted_team: DataFrame containing the predicted team
        actual_team: DataFrame containing the actual optimal team
        
    Returns:
        Dictionary with comparison metrics
    """
    # Debugging information
    print(f"Comparing teams:")
    print(f"  Predicted team shape: {predicted_team.shape}")
    print(f"  Actual team shape: {actual_team.shape}")
    
    # Check if dataframes are empty
    if predicted_team.empty or actual_team.empty:
        print("ERROR: One or both teams are empty!")
        # Return a default comparison with zeros
        return {
            'common_players_count': 0,
            'common_players_percentage': 0,
            'common_players': [],
            'predicted_formation': "0-0-0",
            'actual_formation': "0-0-0",
            'predicted_total': 0,
            'actual_total': 0,
            'point_differential': 0,
            'adjusted_total': 0,
            'adjusted_differential': 0
        }
    
    # Get sets of player names
    predicted_players = set(predicted_team['name'])
    actual_players = set(actual_team['name'])
    
    # Calculate overlap
    common_players = predicted_players.intersection(actual_players)
    
    # Print column names for debugging
    print(f"  Predicted team columns: {predicted_team.columns.tolist()}")
    print(f"  Actual team columns: {actual_team.columns.tolist()}")
    
    # Calculate point differentials with error handling
    predicted_total = None
    actual_total = None
    point_differential = None
    adjusted_total = None
    adjusted_differential = None
    
    # Try to calculate totals safely
    try:
        if 'predicted_points' in predicted_team.columns:
            predicted_total = float(predicted_team['predicted_points'].sum())
            print(f"  Predicted total points: {predicted_total}")
        else:
            print("  WARNING: 'predicted_points' column not found in predicted team")
    except Exception as e:
        print(f"  ERROR calculating predicted total: {e}")
        predicted_total = 0
    
    try:
        if 'total_points' in actual_team.columns:
            actual_total = float(actual_team['total_points'].sum())
            print(f"  Actual total points: {actual_total}")
        else:
            print("  WARNING: 'total_points' column not found in actual team")
    except Exception as e:
        print(f"  ERROR calculating actual total: {e}")
        actual_total = 0
    
    # Calculate differentials only if both totals are valid
    if predicted_total is not None and actual_total is not None:
        try:
            point_differential = actual_total - predicted_total
            print(f"  Point differential: {point_differential}")
        except Exception as e:
            print(f"  ERROR calculating point differential: {e}")
            point_differential = 0
    
    # Calculate adjusted prediction (if available)
    try:
        if 'adjusted_points' in predicted_team.columns:
            adjusted_total = float(predicted_team['adjusted_points'].sum())
            print(f"  Adjusted total points: {adjusted_total}")
            
            if actual_total is not None:
                adjusted_differential = actual_total - adjusted_total
                print(f"  Adjusted differential: {adjusted_differential}")
        else:
            print("  'adjusted_points' column not found in predicted team")
    except Exception as e:
        print(f"  ERROR calculating adjusted points: {e}")
        adjusted_total = None
        adjusted_differential = None
    
    # Get formations with error handling
    try:
        def_count = predicted_team['position_label'].value_counts().get('DEF', 0)
        mid_count = predicted_team['position_label'].value_counts().get('MID', 0)
        fwd_count = predicted_team['position_label'].value_counts().get('FWD', 0)
        predicted_formation = f"{def_count}-{mid_count}-{fwd_count}"
    except Exception as e:
        print(f"  ERROR calculating predicted formation: {e}")
        predicted_formation = "0-0-0"
    
    try:
        actual_def_count = actual_team['position_label'].value_counts().get('DEF', 0)
        actual_mid_count = actual_team['position_label'].value_counts().get('MID', 0)
        actual_fwd_count = actual_team['position_label'].value_counts().get('FWD', 0)
        actual_formation = f"{actual_def_count}-{actual_mid_count}-{actual_fwd_count}"
    except Exception as e:
        print(f"  ERROR calculating actual formation: {e}")
        actual_formation = "0-0-0"
    
    # Ensure values are JSON serializable
    if predicted_total is not None:
        predicted_total = float(predicted_total)
    if actual_total is not None:
        actual_total = float(actual_total)
    if point_differential is not None:
        point_differential = float(point_differential)
    if adjusted_total is not None:
        adjusted_total = float(adjusted_total)
    if adjusted_differential is not None:
        adjusted_differential = float(adjusted_differential)
    
    # Compile results
    results = {
        'common_players_count': len(common_players),
        'common_players_percentage': float(len(common_players) / 11 * 100),
        'common_players': list(common_players),
        'predicted_formation': predicted_formation,
        'actual_formation': actual_formation,
        'predicted_total': predicted_total if predicted_total is not None else 0,
        'actual_total': actual_total if actual_total is not None else 0,
        'point_differential': point_differential if point_differential is not None else 0,
        'adjusted_total': adjusted_total if adjusted_total is not None else 0,
        'adjusted_differential': adjusted_differential if adjusted_differential is not None else 0
    }
    
    return results

# Function to display comparison results
def display_comparison_results(comparison_results: Dict, algorithm: str) -> None:
    """
    Display formatted comparison results
    
    Args:
        comparison_results: Dictionary with comparison metrics
        algorithm: Algorithm name
    """
    print(f"\n===== Comparison Results for {algorithm} =====")
    print(f"Players correctly predicted: {comparison_results['common_players_count']}/11 ({comparison_results['common_players_percentage']:.1f}%)")
    print(f"Common players: {', '.join(comparison_results['common_players'])}")
    print(f"Predicted formation: {comparison_results['predicted_formation']}")
    print(f"Actual formation: {comparison_results['actual_formation']}")
    
    if comparison_results['predicted_total'] is not None:
        print(f"Predicted total points: {comparison_results['predicted_total']:.2f}")
        print(f"Actual total points: {comparison_results['actual_total']:.2f}")
        print(f"Point differential: {comparison_results['point_differential']:.2f}")
        
        if comparison_results['adjusted_total'] is not None:
            print(f"Adjusted total points: {comparison_results['adjusted_total']:.2f}")
            print(f"Adjusted point differential: {comparison_results['adjusted_differential']:.2f}")

# Function to save the best team to a CSV file
def save_team(team_df: pd.DataFrame, gw: int, algorithm: str, team_type: str):
    """
    Save a team to a CSV file
    
    Args:
        team_df: DataFrame containing the team
        gw: Gameweek number
        algorithm: Algorithm name
        team_type: Type of team ('predicted' or 'actual')
    """
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the output directory path
    output_dir = os.path.join(script_dir, "..", "predictions", f"gw{gw:02d}", algorithm, "best_team")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{team_type}_team_{algorithm}_gw{gw:02d}.csv")
    team_df.to_csv(output_file, index=False)
    
    print(f"\nTeam saved to: {output_file}")

# Improved function to save comparison results to a JSON file
def save_comparison_results(comparison_results: Dict, gw: int, algorithm: str):
    """
    Save comparison results to a JSON file
    
    Args:
        comparison_results: Dictionary with comparison metrics
        gw: Gameweek number
        algorithm: Algorithm name
    """
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the output directory path
    output_dir = os.path.join(script_dir, "..", "predictions", f"gw{gw:02d}", algorithm, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"comparison_results_gw{gw:02d}.json")
    
    # Ensure all values are JSON serializable
    serializable_results = {
        'common_players_count': int(comparison_results['common_players_count']),
        'common_players_percentage': float(comparison_results['common_players_percentage']),
        'common_players': list(comparison_results['common_players']),
        'predicted_formation': str(comparison_results['predicted_formation']),
        'actual_formation': str(comparison_results['actual_formation']),
        'predicted_total': float(comparison_results['predicted_total']) if comparison_results['predicted_total'] is not None else 0,
        'actual_total': float(comparison_results['actual_total']) if comparison_results['actual_total'] is not None else 0,
        'point_differential': float(comparison_results['point_differential']) if comparison_results['point_differential'] is not None else 0
    }
    
    # Add adjusted metrics if available
    if comparison_results.get('adjusted_total') is not None:
        serializable_results['adjusted_total'] = float(comparison_results['adjusted_total'])
    if comparison_results.get('adjusted_differential') is not None:
        serializable_results['adjusted_differential'] = float(comparison_results['adjusted_differential'])
    
    try:
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"Comparison results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving comparison results: {e}")
        import traceback
        traceback.print_exc()

# Function to save all algorithm comparisons to a summary CSV
def save_comparison_summary(all_comparisons: Dict, gw: int):
    """
    Save a summary of comparisons across all algorithms
    
    Args:
        all_comparisons: Dictionary mapping algorithm names to comparison results
        gw: Gameweek number
    """
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the output directory path
    output_dir = os.path.join(script_dir, "..", "predictions", f"gw{gw:02d}", "summary")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary DataFrame
    rows = []
    for algorithm, results in all_comparisons.items():
        row = {
            'algorithm': algorithm,
            'gameweek': gw,
            'common_players_count': results['common_players_count'],
            'common_players_percentage': results['common_players_percentage'],
            'predicted_formation': results['predicted_formation'],
            'actual_formation': results['actual_formation'],
            'predicted_total': results['predicted_total'],
            'actual_total': results['actual_total'],
            'point_differential': results['point_differential']
        }
        
        if results['adjusted_total'] is not None:
            row['adjusted_total'] = results['adjusted_total']
            row['adjusted_differential'] = results['adjusted_differential']
        
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    
    # Sort by common players count (descending) and point differential (ascending)
    summary_df = summary_df.sort_values(
        by=['common_players_count', 'point_differential'], 
        ascending=[False, True]
    )
    
    # Save to CSV
    output_file = os.path.join(output_dir, f"algorithm_comparison_summary_gw{gw:02d}.csv")
    summary_df.to_csv(output_file, index=False)
    
    print(f"\nAlgorithm comparison summary saved to: {output_file}")


# Function to save multi-gameweek aggregated summary
def save_multi_gameweek_summary(all_gw_comparisons: Dict):
    """
    Save aggregated summaries from multiple gameweeks
    
    Args:
        all_gw_comparisons: Dictionary mapping gameweeks to algorithm comparison results
    """
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the output directory path
    output_dir = os.path.join(script_dir, "..", "predictions", "summary")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create multi-gameweek summary
    all_rows = []
    
    # Process each gameweek
    for gw, algorithms_data in all_gw_comparisons.items():
        for algorithm, results in algorithms_data.items():
            row = {
                'algorithm': algorithm,
                'gameweek': gw,
                'common_players_count': results['common_players_count'],
                'common_players_percentage': results['common_players_percentage'],
                'predicted_formation': results['predicted_formation'],
                'actual_formation': results['actual_formation']
            }
            
            # Add point values if available
            if results['predicted_total'] is not None:
                row['predicted_total'] = results['predicted_total']
                row['actual_total'] = results['actual_total']
                row['point_differential'] = results['point_differential']
            
            if results['adjusted_total'] is not None:
                row['adjusted_total'] = results['adjusted_total']
                row['adjusted_differential'] = results['adjusted_differential']
            
            all_rows.append(row)
    
    # Create DataFrame with all results
    all_results_df = pd.DataFrame(all_rows)
    
    # Save detailed results
    detail_file = os.path.join(output_dir, "all_gameweeks_detailed.csv")
    all_results_df.to_csv(detail_file, index=False)
    print(f"Detailed multi-gameweek summary saved to: {detail_file}")
    
    # Create algorithm-level summary
    algorithm_summary = all_results_df.groupby('algorithm').agg({
        'common_players_count': ['mean', 'std', 'max', 'min', 'sum'],
        'common_players_percentage': ['mean', 'std'],
        'point_differential': ['mean', 'std', 'min', 'max'] if 'point_differential' in all_results_df.columns else 'count',
        'adjusted_differential': ['mean', 'std', 'min', 'max'] if 'adjusted_differential' in all_results_df.columns else 'count',
        'gameweek': 'count'
    })
    
    # Flatten multi-index columns
    algorithm_summary.columns = ['_'.join(col).strip() for col in algorithm_summary.columns.values]
    algorithm_summary = algorithm_summary.reset_index()
    
    # Rename columns for clarity
    algorithm_summary = algorithm_summary.rename(columns={
        'gameweek_count': 'gameweeks_analyzed',
        'common_players_count_mean': 'avg_common_players',
        'common_players_count_sum': 'total_correct_players',
        'common_players_percentage_mean': 'avg_player_accuracy_pct',
        'point_differential_mean': 'avg_point_differential',
        'adjusted_differential_mean': 'avg_adjusted_differential'
    })
    
    # Sort by average accuracy (descending)
    algorithm_summary = algorithm_summary.sort_values('avg_common_players', ascending=False)
    
    # Save summary file
    summary_file = os.path.join(output_dir, "algorithm_performance_summary.csv")
    algorithm_summary.to_csv(summary_file, index=False)
    print(f"Algorithm performance summary saved to: {summary_file}")
    
    # Create visualization if we have enough data
    if len(all_gw_comparisons) >= 3:
        create_performance_visualization(all_results_df, output_dir)

# Function to create visualizations for multi-gameweek performance
def create_performance_visualization(all_results_df: pd.DataFrame, output_dir: str):
    """
    Create visualizations of algorithm performance across gameweeks
    
    Args:
        all_results_df: DataFrame with all results across gameweeks
        output_dir: Directory to save visualizations
    """
    try:
        # Set up plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create a figure directory
        figure_dir = os.path.join(output_dir, "figures")
        os.makedirs(figure_dir, exist_ok=True)
        
        # 1. Common players percentage over time by algorithm
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=all_results_df, x='gameweek', y='common_players_percentage', 
                    hue='algorithm', marker='o', linewidth=2)
        plt.title('Algorithm Accuracy (% of Correctly Predicted Players) by Gameweek', fontsize=14)
        plt.xlabel('Gameweek', fontsize=12)
        plt.ylabel('Player Accuracy (%)', fontsize=12)
        plt.legend(title='Algorithm', title_fontsize=12, fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, 'player_accuracy_by_gameweek.png'), dpi=150)
        plt.close()
        
        # 2. Point differential boxplot by algorithm
        if 'point_differential' in all_results_df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=all_results_df, x='algorithm', y='point_differential')
            plt.title('Distribution of Point Differential by Algorithm', fontsize=14)
            plt.xlabel('Algorithm', fontsize=12)
            plt.ylabel('Point Differential (Actual - Predicted)', fontsize=12)
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            plt.grid(True, axis='y')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(figure_dir, 'point_differential_distribution.png'), dpi=150)
            plt.close()
        
        # 3. Heatmap of algorithm performance across gameweeks
        common_players_pivot = all_results_df.pivot(index='algorithm', columns='gameweek', 
                                                  values='common_players_count')
        plt.figure(figsize=(12, 8))
        sns.heatmap(common_players_pivot, annot=True, cmap='YlGnBu', fmt='.0f', 
                   linewidths=0.5, cbar_kws={'label': 'Common Players Count'})
        plt.title('Number of Correctly Predicted Players by Algorithm and Gameweek', fontsize=14)
        plt.xlabel('Gameweek', fontsize=12)
        plt.ylabel('Algorithm', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, 'algorithm_gameweek_heatmap.png'), dpi=150)
        plt.close()
        
        # 4. Average performance radar chart for algorithms
        # Only include if we have at least 3 algorithms to compare
        if len(all_results_df['algorithm'].unique()) >= 3:
            from matplotlib.path import Path
            from matplotlib.spines import Spine
            from matplotlib.transforms import Affine2D
            
            # Calculate algorithm average metrics
            algorithm_metrics = all_results_df.groupby('algorithm').agg({
                'common_players_percentage': 'mean',
                'point_differential': 'mean' if 'point_differential' in all_results_df.columns else lambda x: 0
            }).reset_index()
            
            # For point differential, lower is better, so transform to a 0-100 scale
            if 'point_differential' in algorithm_metrics.columns:
                max_diff = algorithm_metrics['point_differential'].max()
                min_diff = algorithm_metrics['point_differential'].min()
                diff_range = max_diff - min_diff
                if diff_range > 0:
                    algorithm_metrics['point_diff_score'] = 100 - ((algorithm_metrics['point_differential'] - min_diff) / diff_range * 100)
                else:
                    algorithm_metrics['point_diff_score'] = 50  # Default if all algorithms have the same differential
            
            # Calculate average formation matching
            formation_match = all_results_df.apply(
                lambda x: 1 if x['predicted_formation'] == x['actual_formation'] else 0, 
                axis=1
            )
            formation_accuracy = formation_match.groupby(all_results_df['algorithm']).mean() * 100
            algorithm_metrics['formation_accuracy'] = algorithm_metrics['algorithm'].map(formation_accuracy)
            
            # Create radar chart
            categories = ['Player Accuracy', 'Point Prediction', 'Formation Accuracy']
            
            # Function to create a radar chart
            def radar_chart(fig, metrics, categories, title):
                # Number of variables
                N = len(categories)
                
                # What will be the angle of each axis
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the loop
                
                # Subplot
                ax = fig.add_subplot(111, polar=True)
                
                # Draw one axis per variable and add labels
                plt.xticks(angles[:-1], categories, fontsize=12)
                
                # Draw ylabels
                ax.set_rlabel_position(0)
                plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], fontsize=8)
                plt.ylim(0, 100)
                
                # Plot each algorithm
                for i, algo in enumerate(metrics['algorithm']):
                    values = [
                        metrics.loc[metrics['algorithm'] == algo, 'common_players_percentage'].values[0],
                        metrics.loc[metrics['algorithm'] == algo, 'point_diff_score'].values[0] if 'point_diff_score' in metrics.columns else 50,
                        metrics.loc[metrics['algorithm'] == algo, 'formation_accuracy'].values[0]
                    ]
                    values += values[:1]  # Close the loop
                    
                    ax.plot(angles, values, linewidth=2, linestyle='solid', label=algo)
                    ax.fill(angles, values, alpha=0.1)
                
                # Add title
                plt.title(title, size=15, y=1.1)
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                
                return ax
            
            # Create the plot
            fig = plt.figure(figsize=(8, 8))
            radar_chart(fig, algorithm_metrics, categories, "Algorithm Performance Metrics")
            plt.tight_layout()
            plt.savefig(os.path.join(figure_dir, 'algorithm_radar_chart.png'), dpi=150)
            plt.close()
        
        print(f"Performance visualizations saved to {figure_dir}")
    
    except Exception as e:
        print(f"Error creating visualizations: {e}")

# Function to process a single gameweek
def process_gameweek(
    gw: int, 
    algorithms: List[str], 
    fpl_data: Dict, 
    args
) -> Dict[str, Dict]:
    """
    Process all operations for a single gameweek
    
    Args:
        gw: Gameweek number
        algorithms: List of algorithms to process
        fpl_data: Dictionary with player fitness data from FPL API
        args: Command-line arguments
        
    Returns:
        Dictionary with comparison results for all algorithms
    """
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Processing Gameweek {gw}")
    print(f"{'='*60}")
    
    # Calculate gameweek for actual data (next gameweek)
    actual_gw = gw + 1
    
    # Dictionary to store comparison results for this gameweek
    all_comparisons = {}
    
    # If not skipping comparison, check if actual gameweek data exists
    actual_data = None
    actual_optimal_team = None
    if not args.no_compare:
        actual_data = load_actual_gameweek_data(actual_gw)
        if actual_data.empty:
            print(f"No actual data found for gameweek {actual_gw}. Comparison will be skipped for GW{gw}.")
        else:
            # Build optimal team from actual data
            print(f"\nBuilding optimal team from actual gameweek {actual_gw} data...")
            try:
                actual_optimal_team = build_optimal_actual_team(
                    actual_data, 
                    pop_size=args.pop_size, 
                    max_generations=args.generations
                )
                if actual_optimal_team.empty:
                    print(f"ERROR: Failed to build actual optimal team for gameweek {actual_gw}")
                else:
                    print(f"Optimal actual team built with {actual_optimal_team['total_points'].sum()} total points")
                    
                    # Display the actual optimal team
                    display_team(actual_optimal_team, f"Actual Optimal Team (GW{actual_gw})")
                    
                    # Save the actual optimal team
                    save_team(actual_optimal_team, gw, "actual", "optimal")
            except Exception as e:
                print(f"ERROR building actual optimal team: {e}")
                import traceback
                traceback.print_exc()
    
    # Skip team optimization if compare-only flag is set
    if not args.compare_only:
        # Process each algorithm
        for algorithm in algorithms:
            print(f"\nProcessing {algorithm} for gameweek {gw}...")
            
            # Load predictions
            predictions_df = load_predictions(gw, algorithm)
            
            if predictions_df.empty:
                print(f"No predictions found for {algorithm} in gameweek {gw}")
                continue
            
            print(f"Loaded {len(predictions_df)} player predictions")
            
            # Merge with FPL data
            merged_df = merge_predictions_with_fpl(predictions_df, fpl_data)
            print("Merged predictions with player fitness data")
            
            # Run genetic algorithm
            print(f"\nRunning genetic algorithm for {algorithm}...")
            ga = GeneticAlgorithm(
                player_pool=merged_df,
                pop_size=args.pop_size,
                max_generations=args.generations,
                stagnation_limit=args.stagnation
            )
            
            best_team_indices = ga.run()
            
            # Get the best team as a DataFrame
            best_team_df = merged_df.iloc[best_team_indices].copy()
            
            # Display the best team
            display_team(best_team_df, f"Predicted Best Team for {algorithm}")
            
            # Save the best team
            save_team(best_team_df, gw, algorithm, "predicted")
    
    # Run comparisons if not skipped and actual data exists
    if not args.no_compare and actual_optimal_team is not None and not actual_optimal_team.empty:
        print("\n\n=== Comparing Predicted Teams with Actual Optimal Team ===\n")
        
        # Process each algorithm
        for algorithm in algorithms:
            # Path to predicted team CSV
            script_dir = os.path.dirname(os.path.abspath(__file__))
            predicted_team_path = os.path.join(script_dir, "..", "predictions", f"gw{gw:02d}", algorithm, "best_team", f"predicted_team_{algorithm}_gw{gw:02d}.csv")
            
            # Load predicted team if it exists
            try:
                print(f"Looking for predicted team at: {predicted_team_path}")
                if os.path.exists(predicted_team_path):
                    predicted_team = pd.read_csv(predicted_team_path)
                    
                    # Compare with actual optimal team
                    print(f"\nComparing {algorithm} predicted team with actual optimal team...")
                    comparison_results = compare_teams(predicted_team, actual_optimal_team)
                    
                    # Display comparison results
                    display_comparison_results(comparison_results, algorithm)
                    
                    # Save comparison results
                    save_comparison_results(comparison_results, gw, algorithm)
                    
                    # Store for summary
                    all_comparisons[algorithm] = comparison_results
                else:
                    print(f"Predicted team file not found for {algorithm} in gameweek {gw}")
            except Exception as e:
                print(f"Error processing comparison for {algorithm}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save summary comparison across all algorithms if we have results
        if all_comparisons:
            save_comparison_summary(all_comparisons, gw)
    
    elapsed_time = time.time() - start_time
    print(f"\nGameweek {gw} processing completed in {elapsed_time:.2f} seconds")
    
    return all_comparisons

# Main function
def main():
    parser = argparse.ArgumentParser(description='FPL Team Optimizer with Ground Truth Comparison (Multi-Gameweek Support)')
    
    # Gameweek selection group (mutually exclusive)
    gw_group = parser.add_mutually_exclusive_group()
    gw_group.add_argument('--gw', type=int, help='Specific gameweek to generate predictions for')
    gw_group.add_argument('--gw-range', type=str, help='Range of gameweeks to process (e.g., "1-10")')
    gw_group.add_argument('--gws', type=str, help='Comma-separated list of gameweeks to process (e.g., "1,3,5,7")')
    gw_group.add_argument('--all-gws', action='store_true', help='Process all available gameweeks')
    
    # Algorithm selection
    parser.add_argument('--algorithms', nargs='+', help='Learning algorithms to use (default: all)')
    
    # Genetic algorithm settings
    parser.add_argument('--pop-size', type=int, default=100, help='Population size for genetic algorithm')
    parser.add_argument('--generations', type=int, default=1000, help='Maximum number of generations for genetic algorithm')
    parser.add_argument('--stagnation', type=int, default=100, help='Stop after this many generations without improvement')
    
    # Processing options
    parser.add_argument('--compare-only', action='store_true', help='Skip optimization and only run comparison with existing teams')
    parser.add_argument('--no-compare', action='store_true', help='Skip comparison with actual teams')
    parser.add_argument('--parallel', action='store_true', help='Process multiple gameweeks in parallel')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel workers (default: 4)')
    
    args = parser.parse_args()
    
    # Default algorithms if not specified
    if not args.algorithms:
        args.algorithms = ['DeepLearning', 'GBReg', 'LSTM', 'RFReg', 'Ridge', 'XGBoost']
    
    # Print selected algorithms
    print(f"Selected algorithms: {args.algorithms}")
    
    # Determine which gameweeks to process
    gameweeks_to_process = []
    
    if args.gw:
        # Single gameweek
        gameweeks_to_process = [args.gw]
    elif args.gw_range:
        # Range of gameweeks
        try:
            start_gw, end_gw = map(int, args.gw_range.split('-'))
            gameweeks_to_process = list(range(start_gw, end_gw + 1))
        except ValueError:
            print(f"Error parsing gameweek range: {args.gw_range}")
            print("Format should be 'start-end', e.g., '1-10'")
            return
    elif args.gws:
        # List of specific gameweeks
        try:
            gameweeks_to_process = [int(gw.strip()) for gw in args.gws.split(',')]
        except ValueError:
            print(f"Error parsing gameweek list: {args.gws}")
            print("Format should be comma-separated values, e.g., '1,3,5,7'")
            return
    elif args.all_gws:
        # All available gameweeks
        gameweeks_to_process = find_available_gameweeks(args.algorithms)
        if not gameweeks_to_process:
            print("No gameweeks found with prediction data for the specified algorithms.")
            # Print the current directory structure for debugging
            script_dir = os.path.dirname(os.path.abspath(__file__))
            pred_dir = os.path.join(script_dir, "..", "predictions")
            if os.path.exists(pred_dir):
                print(f"Contents of predictions directory: {os.listdir(pred_dir)}")
                # Check a few potential gameweek directories
                for i in range(1, 6):
                    gw_dir = os.path.join(pred_dir, f"gw{i:02d}")
                    if os.path.exists(gw_dir):
                        print(f"Contents of {gw_dir}: {os.listdir(gw_dir)}")
            return
    else:
        # Default to the latest available gameweek
        available_gws = find_available_gameweeks(args.algorithms)
        if not available_gws:
            print("No gameweeks found with prediction data. Please run the prediction script first.")
            return
        gameweeks_to_process = [max(available_gws)]
    
    # Sort gameweeks
    gameweeks_to_process.sort()
    
    print(f"Will process gameweeks: {gameweeks_to_process}")
    
    # Fetch FPL data once (player fitness)
    print("\nFetching player fitness data from FPL API...")
    fpl_data = fetch_fpl_data()
    
    # Process gameweeks
    all_gw_comparisons = {}
    
    # Determine whether to use parallel processing
    if args.parallel and len(gameweeks_to_process) > 1:
        print(f"Processing {len(gameweeks_to_process)} gameweeks in parallel with {args.max_workers} workers...")
        
        # Use concurrent.futures to process gameweeks in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all gameweeks for processing
            future_to_gw = {
                executor.submit(process_gameweek, gw, args.algorithms, fpl_data, args): gw 
                for gw in gameweeks_to_process
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_gw):
                gw = future_to_gw[future]
                try:
                    comparisons = future.result()
                    if comparisons:
                        all_gw_comparisons[gw] = comparisons
                        print(f"Gameweek {gw} processed successfully with {len(comparisons)} algorithm comparisons")
                except Exception as e:
                    print(f"Error processing gameweek {gw}: {e}")
    else:
        # Process gameweeks sequentially
        for gw in gameweeks_to_process:
            try:
                comparisons = process_gameweek(gw, args.algorithms, fpl_data, args)
                if comparisons:
                    all_gw_comparisons[gw] = comparisons
                    print(f"Gameweek {gw} processed successfully with {len(comparisons)} algorithm comparisons")
            except Exception as e:
                print(f"Error processing gameweek {gw}: {e}")
                import traceback
                traceback.print_exc()
    
    # Debug output for all_gw_comparisons
    print(f"\nCollected comparison data for {len(all_gw_comparisons)} gameweeks")
    for gw, algos in all_gw_comparisons.items():
        print(f"  Gameweek {gw}: {len(algos)} algorithm comparisons")
    
    # Generate multi-gameweek summary if we processed more than one gameweek
    if len(all_gw_comparisons) > 1:
        print("\n\n=== Generating Multi-Gameweek Performance Summary ===\n")
        save_multi_gameweek_summary(all_gw_comparisons)
    elif len(all_gw_comparisons) == 1:
        print("\nOnly one gameweek with comparisons, skipping multi-gameweek summary")
    else:
        print("\nNo gameweeks with valid comparisons, cannot generate summary")
    
    print("\nAll processing completed!")

if __name__ == "__main__":
    main()
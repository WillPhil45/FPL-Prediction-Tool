import pandas as pd
import numpy as np
import os
import argparse

# Add command line argument parsing
parser = argparse.ArgumentParser(description='Generate prediction datasets for multiple gameweeks')
parser.add_argument('--max_gw', type=int, required=True, help='Most recent gameweek to process')
args = parser.parse_args()

# Define constants
MAX_GW = args.max_gw
BASE_DIR = r"Up-to-data\Fantasy-Premier-League\data\2024-25\gws"
MERGED_PATH = os.path.join(r"data\merged_gw_clean.csv")
FIXTURES_PATH = os.path.join("Up-to-data", "Fantasy-Premier-League", "data", "2024-25", "fixtures.csv")

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

TRAINING_COLS = [
    "name","assists","bonus","bps","clean_sheets","creativity","element","goals_conceded","goals_scored","ict_index","influence",
    "minutes","opponent_team","own_goals","penalties_missed","penalties_saved","red_cards","round","saves","selected",
    "team_a_score","team_h_score","threat","current_match_points","transfers_balance","transfers_in","transfers_out","value","was_home","yellow_cards",
    "GW","assists_last_game","total_assists_season","avg_assists_last_3","avg_assists_last_10","bonus_last_game","total_bonus_season",
    "avg_bonus_last_3","avg_bonus_last_10","bps_last_game","total_bps_season","avg_bps_last_3","avg_bps_last_10","clean_sheets_last_game",
    "total_clean_sheets_season","avg_clean_sheets_last_3","avg_clean_sheets_last_10","goals_conceded_last_game","total_goals_conceded_season",
    "avg_goals_conceded_last_3","avg_goals_conceded_last_10","goals_scored_last_game","total_goals_scored_season","avg_goals_scored_last_3",
    "avg_goals_scored_last_10","own_goals_last_game","total_own_goals_season","avg_own_goals_last_3","avg_own_goals_last_10",
    "penalties_missed_last_game","total_penalties_missed_season","avg_penalties_missed_last_3","avg_penalties_missed_last_10",
    "penalties_saved_last_game","total_penalties_saved_season","avg_penalties_saved_last_3","avg_penalties_saved_last_10",
    "red_cards_last_game","total_red_cards_season","avg_red_cards_last_3","avg_red_cards_last_10","saves_last_game","total_saves_season",
    "avg_saves_last_3","avg_saves_last_10","yellow_cards_last_game","total_yellow_cards_season","avg_yellow_cards_last_3","avg_yellow_cards_last_10",
    "minutes_last_game","total_minutes_season","avg_minutes_last_3","avg_minutes_last_10","form","games_played",
    "cumulative_points","ppm","was_home_last_game","position_DEF","position_FWD","position_GK","position_MID","fixture_difficulty",
    "has_fixture_next_week",  # Added new column
]

LAG_FEATURES = [
    "assists","bonus","bps","clean_sheets","goals_conceded","goals_scored","own_goals",
    "penalties_missed","penalties_saved","red_cards","saves","yellow_cards","minutes"
]

def build_rolling_features(group: pd.DataFrame):
    # shift-based rolling for LAG_FEATURES
    for feat in LAG_FEATURES:
        group[f"{feat}_last_game"] = group[feat].shift(1).fillna(0)
        group[f"total_{feat}_season"] = group[feat].cumsum().shift(1).fillna(0)
        group[f"avg_{feat}_last_3"] = (
            group[feat].rolling(3, min_periods=1).mean().shift(1).fillna(0)
        )
        group[f"avg_{feat}_last_10"] = (
            group[feat].rolling(10, min_periods=1).mean().shift(1).fillna(0)
        )
    # form from current_match_points
    if 'current_match_points' not in group.columns:
        group['current_match_points'] = 0
    group['form'] = (
        group['current_match_points']
        .rolling(5, min_periods=1)
        .mean()
        .shift(1)
        .fillna(0)
    )
    # minutes_played, games_played
    group['games_played'] = (group['minutes'] > 0).cumsum()
    # cumsum => cumulative_points => ppm
    group['cumulative_points'] = group['current_match_points'].cumsum().shift(1).fillna(0)
    group['ppm'] = group['cumulative_points'] / group['games_played']
    group['ppm'] = group['ppm'].fillna(0)

    # was_home_last_game: shift(1) from was_home
    group['was_home_last_game'] = group['was_home'].shift(1).fillna(0).astype(int)
    return group

def process_gameweek(gw):
    print(f"\n\n=== Processing Gameweek {gw} ===")
    gw_str = str(gw)
    OUT_FILE = os.path.join(OUT_DIR, f"player_predictions_data_gw{gw_str.zfill(2)}.csv")
    
    GW_PATH = os.path.join(BASE_DIR, "gw" + gw_str + ".csv")
    
    # Check if the gameweek file exists
    if not os.path.exists(GW_PATH):
        print(f"Could not find {GW_PATH}, skipping gameweek {gw}")
        return False
    
    # ----------------------------------------------------------------
    # 1) Load current gameweek data
    # ----------------------------------------------------------------
    df_gw = pd.read_csv(GW_PATH, low_memory=True)

    # If file uses "total_points" as the column name:
    if 'total_points' in df_gw.columns and 'current_match_points' not in df_gw.columns:
        df_gw.rename(columns={'total_points':'current_match_points'}, inplace=True)

    # If there's no 'was_home' column, fill it with 1 or 0 as default
    if 'was_home' not in df_gw.columns:
        df_gw['was_home'] = df_gw.get('home', False).astype(bool).astype(int)

    # ----------------------------------------------------------------
    # 2) Load merged_gw_clean for all gameweeks before the current one
    # ----------------------------------------------------------------
    if not os.path.exists(MERGED_PATH):
        print(f"Could not find {MERGED_PATH}, skipping gameweek {gw}")
        return False
    
    df_merged = pd.read_csv(MERGED_PATH, low_memory=True)
    df_merged = df_merged[df_merged['round'] < gw].copy()

    # Standard rename if "total_points" is present
    if 'total_points' in df_merged.columns and 'current_match_points' not in df_merged.columns:
        df_merged.rename(columns={'total_points':'current_match_points'}, inplace=True)

    # If 'was_home' is absent, define a dummy
    if 'was_home' not in df_merged.columns:
        df_merged['was_home'] = df_merged.get('home', False).astype(bool).astype(int)

    # ----------------------------------------------------------------
    # 3) Unify on 'element' to identify players
    # ----------------------------------------------------------------
    if 'element' not in df_merged.columns:
        df_merged['element'] = df_merged['name'].astype('category').cat.codes
    if 'element' not in df_gw.columns:
        df_gw['element'] = df_gw['name'].astype('category').cat.codes

    # ----------------------------------------------------------------
    # 4) Combine older GWs with current GW and build rolling stats
    # ----------------------------------------------------------------
    df_all = pd.concat([df_merged, df_gw], ignore_index=True)
    df_all.sort_values(by=['element','round'], inplace=True, ignore_index=True)

    df_all = df_all.groupby('element', as_index=False, group_keys=False).apply(build_rolling_features)

    # Now we only want the row from the current gameweek for each player
    df_recent = df_all[df_all['round'] == gw].copy()
    df_recent.rename(columns={'team':'team_x'}, inplace=True)

    # ----------------------------------------------------------------
    # 5) fixture_difficulty & has_fixture_next_week
    # ----------------------------------------------------------------
    # Initialize has_fixture_next_week to 0 (no fixture by default)
    df_recent['has_fixture_next_week'] = 0
    df_recent['fixture_difficulty'] = np.nan
    
    if not os.path.exists(FIXTURES_PATH):
        print(f"[Warning] No {FIXTURES_PATH}, fixture_difficulty=NaN and has_fixture_next_week=0.")
    else:
        fx = pd.read_csv(FIXTURES_PATH, low_memory=True)
        
        # We'll isolate next week's matches
        next_gw = gw + 1
        df_next = fx[fx['event'] == next_gw].copy()
        
        if df_next.empty:
            print(f"[Warning] No fixtures found for gameweek {next_gw}, fixture_difficulty=NaN and has_fixture_next_week=0.")
        else:
            # Build a map of {team_a -> team_a_difficulty, team_h -> team_h_difficulty}
            # and track which teams have fixtures next week
            next_diff_map = {}
            teams_with_fixtures = set()  # Track teams with fixtures
            
            for _, rowf in df_next.iterrows():
                ta = rowf['team_a']
                th = rowf['team_h']
                ta_diff = rowf['team_a_difficulty']
                th_diff = rowf['team_h_difficulty']
                
                # Store difficulties
                next_diff_map[ta] = ta_diff
                next_diff_map[th] = th_diff
                
                # Mark teams as having fixtures
                teams_with_fixtures.add(ta)
                teams_with_fixtures.add(th)

            team_list_path = r'data\teams2024-25.csv'
            if os.path.exists(team_list_path):
                tm_df = pd.read_csv(team_list_path)
                df_recent = pd.merge(
                    df_recent,
                    tm_df,
                    left_on=['team_x'],
                    right_on=['team_name'],
                    how='left'
                )

                if 'id' not in df_recent.columns:
                    print("[Warning] df_recent has no 'id' column, cannot map fixture_difficulty.")
                else:
                    # Update has_fixture_next_week based on teams with fixtures
                    df_recent['has_fixture_next_week'] = df_recent['id'].apply(
                        lambda tid: 1 if tid in teams_with_fixtures else 0
                    )
                    
                    # Set fixture_difficulty for teams with fixtures, leave as NaN for others
                    def get_next_diff(tid):
                        return next_diff_map.get(tid, np.nan)
                    
                    df_recent['fixture_difficulty'] = df_recent['id'].apply(get_next_diff)
                    
                    # Fill NaN fixture_difficulty with average difficulty for teams with fixtures
                    # Only for teams that have fixtures
                    avg_difficulty = np.mean(list(next_diff_map.values()))
                    mask = (df_recent['has_fixture_next_week'] == 1) & (df_recent['fixture_difficulty'].isna())
                    df_recent.loc[mask, 'fixture_difficulty'] = avg_difficulty
                    
                    # For teams without fixtures, leave difficulty as NaN
            else:
                print(f"[Warning] No {team_list_path} found, fixture_difficulty=NaN and has_fixture_next_week=0.")

    # If 'value' is in base points (like 50 means 5.0)
    if 'value' in df_recent.columns:
        df_recent['value'] = df_recent['value'] / 10.0

    # ----------------------------------------------------------------
    # 6) position_{GK,DEF,MID,FWD} if 'position' is string-based
    # ----------------------------------------------------------------
    if 'position' in df_recent.columns:
        df_recent['position_GK']  = (df_recent['position'] == 'GK').astype(int)
        df_recent['position_DEF'] = (df_recent['position'] == 'DEF').astype(int)
        df_recent['position_MID'] = (df_recent['position'] == 'MID').astype(int)
        df_recent['position_FWD'] = (df_recent['position'] == 'FWD').astype(int)

    # ----------------------------------------------------------------
    # 7) Drop extraneous columns, fill missing, reorder
    # ----------------------------------------------------------------
    extra_cols = set(df_recent.columns) - set(TRAINING_COLS)
    df_recent.drop(columns=extra_cols, inplace=True, errors='ignore')

    missing_cols = [c for c in TRAINING_COLS if c not in df_recent.columns]
    for col in missing_cols:
        df_recent[col] = np.nan

    # Make sure has_fixture_next_week is an integer
    if 'has_fixture_next_week' in df_recent.columns:
        df_recent['has_fixture_next_week'] = df_recent['has_fixture_next_week'].fillna(0).astype(int)

    # Ensure all columns listed in TRAINING_COLS are present
    for col in TRAINING_COLS:
        if col not in df_recent.columns:
            df_recent[col] = np.nan if col != 'has_fixture_next_week' else 0

    df_recent = df_recent[TRAINING_COLS]

    df_recent['GW'] = gw

    # Debug: Check if these columns have data
    print("\n[DEBUG] Checking columns of interest:")
    check_cols = ['was_home_last_game', 'fixture_difficulty', 'has_fixture_next_week', 'current_match_points']
    for ccc in check_cols:
        if ccc in df_recent.columns:
            print(f" -> {ccc}: Non-null count={df_recent[ccc].notnull().sum()}  Unique={df_recent[ccc].unique()[:10]}")
        else:
            print(f" -> {ccc} not in DataFrame at all")

    # Count players with/without fixtures
    has_fixture_count = df_recent['has_fixture_next_week'].sum()
    no_fixture_count = len(df_recent) - has_fixture_count
    print(f"\nPlayers with fixtures next week: {has_fixture_count}")
    print(f"Players without fixtures next week: {no_fixture_count}")

    # ----------------------------------------------------------------
    # 8) Save final
    # ----------------------------------------------------------------
    df_recent.to_csv(OUT_FILE, index=False)
    print(f"[Info] Saved predictions for GW {gw} to {OUT_FILE}, shape={df_recent.shape}")
    return True

def main():
    successful_gws = 0
    skipped_gws = 0
    
    # Process each gameweek from the most recent down to gameweek 1
    for gw in range(MAX_GW, 0, -1):
        if process_gameweek(gw):
            successful_gws += 1
        else:
            skipped_gws += 1
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed {successful_gws} gameweeks")
    print(f"Skipped {skipped_gws} gameweeks")

if __name__ == "__main__":
    main()
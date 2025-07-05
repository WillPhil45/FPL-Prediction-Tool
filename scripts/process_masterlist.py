import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------
# 0) Read CSV and remove 2016–17, 2017–18 (no fixtures)
# ------------------------------------------------------------
df = pd.read_csv(r'Up-to-data\Fantasy-Premier-League\data\cleaned_merged_seasons.csv', low_memory=True)
df = df[~df['season_x'].isin(['2016-17', '2017-18'])]

df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], errors='coerce')
df['game_date'] = df['kickoff_time'].dt.date

# Ensure we have numeric round (if it's not numeric, convert or fill)
df['round'] = pd.to_numeric(df['round'], errors='coerce').fillna(0)

# Combine GKP -> GK
df['position'] = df['position'].replace({'GKP': 'GK'})

# ------------------------------------------------------------
# 1) Create lagged & cumulative features (similar to your original script)
# ------------------------------------------------------------
lagged_features = [
    'assists', 'bonus', 'bps', 'clean_sheets', 'goals_conceded',
    'goals_scored', 'own_goals', 'penalties_missed', 'penalties_saved',
    'red_cards', 'saves', 'yellow_cards', 'minutes'
]

for feature in lagged_features:
    df[f'{feature}_last_game'] = (
        df.groupby(['name', 'season_x'])[feature]
          .shift(1)
          .fillna(0)
    )
    df[f'total_{feature}_season'] = (
        df.groupby(['name', 'season_x'])[feature]
          .cumsum()
          .shift(1)
          .fillna(0)
    )
    df[f'avg_{feature}_last_3'] = (
        df.groupby(['name', 'season_x'])[feature]
          .rolling(window=3, min_periods=1)
          .mean()
          .shift(1)
          .fillna(0)
          .reset_index(level=[0, 1], drop=True)
    )
    df[f'avg_{feature}_last_10'] = (
        df.groupby(['name', 'season_x'])[feature]
          .rolling(window=10, min_periods=1)
          .mean()
          .shift(1)
          .fillna(0)
          .reset_index(level=[0, 1], drop=True)
    )

# ------------------------------------------------------------
# 2) "Form" as rolling avg of last 5 matches' points
# ------------------------------------------------------------
df['form'] = (
    df.groupby(['name', 'season_x'])['total_points']
      .rolling(window=5, min_periods=1)
      .mean()
      .shift(1)
      .fillna(0)
      .reset_index(level=[0, 1], drop=True)
)

# ------------------------------------------------------------
# 3) Minutes played & games played (cumulative)
# ------------------------------------------------------------
df['games_played'] = (
    df.groupby(['name', 'season_x'])['minutes']
      .transform(lambda x: (x > 0).cumsum())
)

# ------------------------------------------------------------
# 4) SHIFT the next match label & minutes
# ------------------------------------------------------------
df['next_match_points'] = df.groupby(['name','season_x'])['total_points'].shift(-1)
df['next_match_minutes'] = df.groupby(['name','season_x'])['minutes'].shift(-1)
df.dropna(subset=['next_match_points','next_match_minutes'], inplace=True)

# Rename so old total_points doesn't confuse the new label
df.rename(columns={'total_points': 'current_match_points'}, inplace=True)

# ------------------------------------------------------------
# 5) Create "played_next_match" & "started_next_match"
# ------------------------------------------------------------
df['played_next_match'] = (df['next_match_minutes'] > 0).astype(int)
df['started_next_match'] = (df['next_match_minutes'] >= 60).astype(int)

# Cumulative points (prior to this match)
df['cumulative_points'] = (
    df.groupby(['name', 'season_x'])['current_match_points']
      .cumsum()
      .shift(1)
      .fillna(0)
)
df['ppm'] = df['cumulative_points'] / df['games_played']
df['ppm'] = df['ppm'].fillna(0)

# ------------------------------------------------------------
# 6) SHIFT fixture to next_fixture
# ------------------------------------------------------------
df.rename(columns={'fixture': 'current_fixture'}, inplace=True)
df['next_fixture'] = df.groupby(['name','season_x'])['current_fixture'].shift(-1)
df.dropna(subset=['next_fixture'], inplace=True)

# ------------------------------------------------------------
# 7) was_home, was_home_last_game
# ------------------------------------------------------------
df['was_home_last_game'] = (
    df.groupby(['name', 'season_x'])['was_home']
      .shift(1)
      .fillna(0)
      .astype(int)
)
df['was_home'] = df['was_home'].astype(int)

# Convert position to dummies
df = pd.get_dummies(df, columns=['position'], prefix='position')
df['value'] = df['value'] / 10.0

for col in ['position_DEF','position_FWD','position_GK','position_MID']:
    if col in df.columns:
        df[col] = df[col].astype(int)

# ------------------------------------------------------------
# 8) Merge fixture data (same as your original code)
# ------------------------------------------------------------
fixture_cols = ['id','team_a','team_a_difficulty','team_h','team_h_difficulty']
fixtures_list = []
possible_seasons = [
    '2018-19','2019-20','2020-21','2021-22','2022-23','2023-24','2024-25'
]

for s in possible_seasons:
    fix_path = f'Up-to-data/Fantasy-Premier-League/data/{s}/fixtures.csv'
    if os.path.exists(fix_path):
        fixdf = pd.read_csv(fix_path, low_memory=True)
        fixdf = fixdf[fixture_cols].copy()
        fixdf['season_x'] = s
        fixtures_list.append(fixdf)

if fixtures_list:
    fixtures_master = pd.concat(fixtures_list, ignore_index=True)
    fixtures_master.rename(columns={'id':'fixture'}, inplace=True)
    df = pd.merge(
        df,
        fixtures_master,
        left_on=['season_x','next_fixture'],
        right_on=['season_x','fixture'],
        how='left'
    )
else:
    print("No fixture data found for 2018–25 seasons!")

# ------------------------------------------------------------
# 9) Merge with master_team_list => fixture difficulty
# ------------------------------------------------------------
team_list_path = r'Up-to-data\Fantasy-Premier-League\data\master_team_list.csv'
if os.path.exists(team_list_path):
    tm_df = pd.read_csv(team_list_path)
    df = pd.merge(
        df,
        tm_df,
        left_on=['season_x','team_x'],
        right_on=['season','team_name'],
        how='left'
    )
    # Determine fixture difficulty
    df['fixture_difficulty'] = np.where(
        df['team'] == df['team_a'],
        df['team_a_difficulty'],
        df['team_h_difficulty']
    )
else:
    print("No master_team_list.csv found; can't map team_x to numeric ID.")

df.dropna(subset=['fixture_difficulty'], how='any', inplace=True)

# ------------------------------------------------------------
# 10) Clean numeric columns
# ------------------------------------------------------------
def clean_data(df_in):
    numeric_cols = df_in.select_dtypes(include=[np.number]).columns
    df_in[numeric_cols] = df_in[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df_in[numeric_cols] = df_in[numeric_cols].fillna(0)
    return df_in

df = clean_data(df)

# ------------------------------------------------------------
# 11) Remove columns no longer needed
# ------------------------------------------------------------
drop_columns = [
    'current_fixture','fixture','team_a','team_h','team_a_difficulty','team_h_difficulty',
    'team_y','team_name','season','id','next_match_minutes', 'season_x','game_date','kickoff_time'
]
drop_columns = list(set(drop_columns))
df.drop(columns=drop_columns, inplace=True, errors='ignore')

# ------------------------------------------------------------
# 12) Time-Based Split => Train (70%), Val (10%), Test (20%)
# ------------------------------------------------------------
total_len = len(df)
train_end = int(0.70 * total_len)
val_end = int(0.80 * total_len)

df = df.sample(frac=1).reset_index(drop=True)

train_df = df.iloc[:train_end].copy()
val_df   = df.iloc[train_end:val_end].copy()
test_df  = df.iloc[val_end:].copy()

# ------------------------------------------------------------
# 13) Time-Based Sorting
# Sort by (GW)
# ------------------------------------------------------------

train_df = train_df.sort_values(by=['GW']).reset_index(drop=True)
val_df = val_df.sort_values(by=['GW']).reset_index(drop=True)
test_df = test_df.sort_values(by=['GW']).reset_index(drop=True)

print("Time-based split sizes:")
print(f"  Train: {len(train_df)} rows")
print(f"  Val:   {len(val_df)} rows")
print(f"  Test:  {len(test_df)} rows")

# ------------------------------------------------------------
# 14) Save final training, validation, & testing sets
# ------------------------------------------------------------
os.makedirs('data', exist_ok=True)
train_df.to_csv('data/time_train_data.csv', index=False)
val_df.to_csv('data/time_val_data.csv', index=False)
test_df.to_csv('data/time_test_data.csv', index=False)

print("Time-based train/val/test datasets saved successfully!")

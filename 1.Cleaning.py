import pandas as pd
import numpy as np
import os
from datetime import timedelta
ball_path = r"C:\Users\sakha\Downloads\PythonProject\IPL_Ball_by_Ball_2008_2022.csv"
match_path = r"C:\Users\sakha\Downloads\PythonProject\IPL_Matches_2008_2022.csv"
if not os.path.exists(ball_path):
    print(f"Error: Ball file not found at {ball_path}")
    print("Directory contents:", os.listdir(r"C:\Users\sakha\Downloads\PythonProject"))
    exit(1)
if not os.path.exists(match_path):
    print(f"Error: Match file not found at {match_path}")
    print("Directory contents:", os.listdir(r"C:\Users\sakha\Downloads\PythonProject"))
    exit(1)
try:
    print(f"Loading ball file: {ball_path}")
    ball_df = pd.read_csv(ball_path, low_memory=False)
    print("Ball DataFrame Shape:", ball_df.shape)
    print("\nTop 5 rows of Ball dataset:")
    print(ball_df.head())
except Exception as e:
    print(f"Error loading ball file: {e}")
    exit(1)
try:
    print(f"\nLoading match file: {match_path}")
    match_df = pd.read_csv(match_path, low_memory=False)
    print("Match DataFrame Shape:", match_df.shape)
    print("\nTop 5 rows of Match dataset:")
    print(match_df.head())
except Exception as e:
    print(f"Error loading match file: {e}")
    exit(1)
ball_df.replace('NA', np.nan, inplace=True)
if 'ID' in ball_df.columns:
    ball_df['ID'] = ball_df['ID'].ffill()
def fill_innings_na(df, col='innings'):
    df[col] = df[col].astype(float)
    for idx in df[df[col].isna()].index:
        prev_val = df.at[idx-1, col] if idx > 0 else np.nan
        next_val = df.at[idx+1, col] if idx < len(df)-1 else np.nan
        if prev_val == next_val and not pd.isna(prev_val):
            df.at[idx, col] = prev_val
        else:
            prev_2 = [df.at[idx-i, col] for i in range(1, 3) if idx-i >= 0]
            next_2 = [df.at[idx+i, col] for i in range(1, 3) if idx+i < len(df)]
            if len(set(prev_2)) == 1 and not pd.isna(prev_2[0]):
                df.at[idx, col] = prev_2[0]
            elif not pd.isna(next_val):
                df.at[idx, col] = next_val
    return df
if 'innings' in ball_df.columns:
    ball_df = fill_innings_na(ball_df, 'innings')
if 'overs' in ball_df.columns:
    ball_df = fill_innings_na(ball_df, 'overs')
if 'ballnumber' in ball_df.columns:
    ball_df['ballnumber'] = ball_df['ballnumber'].astype(float)
    for idx in ball_df[ball_df['ballnumber'].isna()].index:
        if idx > 0:
            ball_df.at[idx, 'ballnumber'] = ball_df.at[idx-1, 'ballnumber'] + 1
if 'batter' in ball_df.columns and 'bowler' in ball_df.columns:
    init_rows = len(ball_df)
    ball_df = ball_df.dropna(subset=['batter', 'bowler'])
    print(f"Removed {init_rows - len(ball_df)} rows due to NA in batter or bowler")
run_cols = ['batsman_run', 'extras_run', 'total_run']
if all(col in ball_df.columns for col in run_cols):
    for idx in ball_df[ball_df[run_cols].isna().any(axis=1)].index:
        row = ball_df.loc[idx, run_cols]
        if pd.isna(row['total_run']) and not pd.isna(row['batsman_run']) and not pd.isna(row['extras_run']):
            ball_df.at[idx, 'total_run'] = row['batsman_run'] + row['extras_run']
        elif pd.isna(row['batsman_run']) and not pd.isna(row['total_run']) and not pd.isna(row['extras_run']):
            ball_df.at[idx, 'batsman_run'] = row['total_run'] - row['extras_run']
        elif pd.isna(row['extras_run']) and not pd.isna(row['total_run']) and not pd.isna(row['batsman_run']):
            ball_df.at[idx, 'extras_run'] = row['total_run'] - row['batsman_run']
print("\nBall Missing Values After NA Handling:")
print(ball_df.isnull().sum())
match_df.replace('NA', np.nan, inplace=True)
if 'ID' in match_df.columns:
    match_df['ID'] = match_df['ID'].ffill()
if 'City' in match_df.columns:
    match_df['City'] = match_df['City'].replace('Bangalore', 'Bengaluru')
    match_df['City'] = match_df['City'].fillna('Dubai')
if 'Date' in match_df.columns:
    match_df['Date'] = pd.to_datetime(match_df['Date'], errors='coerce')
    for idx in match_df[match_df['Date'].isna()].index:
        if idx > 0:
            match_df.at[idx, 'Date'] = match_df.at[idx-1, 'Date'] + timedelta(days=1)
if 'MatchNumber' in match_df.columns:
    match_df['MatchNumber'] = pd.to_numeric(match_df['MatchNumber'], errors='coerce')
    for idx in match_df[match_df['MatchNumber'].isna()].index:
        if idx > 0:
            match_df.at[idx, 'MatchNumber'] = match_df.at[idx-1, 'MatchNumber'] + 1
if 'Season' in match_df.columns:
    match_df['Season'] = match_df['Season'].ffill()
    season_map = {'2007/08': '2008', '2009/10': '2010', '2020/21': '2020'}
    match_df['Season'] = match_df['Season'].replace(season_map)
if 'Team1' in match_df.columns and 'Team2' in match_df.columns:
    init_rows = len(match_df)
    match_df = match_df.dropna(subset=['Team1', 'Team2'])
    print(f"Removed {init_rows - len(match_df)} rows due to NA in Team1 or Team2")
if 'SuperOver' in match_df.columns and 'Margin' in match_df.columns:
    abandoned_mask = match_df['SuperOver'].isna() & match_df['Margin'].isna()
    if 'WinningTeam' in match_df.columns:
        match_df.loc[abandoned_mask, 'WinningTeam'] = 'Abandoned'
    super_over_mask = match_df['Margin'].isna() & ~match_df['SuperOver'].isna()
    match_df.loc[super_over_mask, 'Margin'] = 0
if 'Player_of_Match' in match_df.columns and 'WinningTeam' in match_df.columns:
    match_df.loc[match_df['Player_of_Match'].isna(), 'WinningTeam'] = 'Abandoned'
print("\nMatch Missing Values After NA Handling:")
print(match_df.isnull().sum())
valid_teams = set(match_df['Team1']).union(set(match_df['Team2'])).union(set(ball_df.get('BattingTeam', [])))
def standardize_team_name(name):
    if pd.isna(name):
        return 'Unknown'
    name = str(name).strip().title()
    return name if name in valid_teams else 'Unknown'
if 'BattingTeam' in ball_df.columns:
    ball_df['BattingTeam'] = ball_df['BattingTeam'].apply(standardize_team_name)
match_df['Team1'] = match_df['Team1'].apply(standardize_team_name)
match_df['Team2'] = match_df['Team2'].apply(standardize_team_name)
match_df['TossWinner'] = match_df['TossWinner'].apply(standardize_team_name)
if 'WinningTeam' in match_df.columns:
    match_df['WinningTeam'] = match_df['WinningTeam'].apply(lambda x: x if x in valid_teams or x in ['No Result', 'Abandoned'] else 'Unknown')
def standardize_player_name_upper(name):
    if pd.isna(name) or name in ['none', 'None', 'No Result']:
        return 'NONE'
    return str(name).strip().upper()
player_cols = ['batter', 'bowler', 'non-striker', 'player_out', 'fielders_involved']
for col in player_cols:
    if col in ball_df.columns:
        ball_df[col] = ball_df[col].apply(standardize_player_name_upper)
if 'Player_of_Match' in match_df.columns:
    match_df['Player_of_Match'] = match_df['Player_of_Match'].apply(standardize_player_name_upper)
if 'Team1Players' in match_df.columns:
    match_df['Team1Players'] = match_df['Team1Players'].str.strip("[]").str.replace("'", "").str.split(", ")
    match_df['Team1Players'] = match_df['Team1Players'].apply(lambda x: [standardize_player_name_upper(name) for name in x] if isinstance(x, list) else ['NONE'])
if 'Team2Players' in match_df.columns:
    match_df['Team2Players'] = match_df['Team2Players'].str.strip("[]").str.replace("'", "").str.split(", ")
    match_df['Team2Players'] = match_df['Team2Players'].apply(lambda x: [standardize_player_name_upper(name) for name in x] if isinstance(x, list) else ['NONE'])
num_cols_ball = ['ID', 'innings', 'overs', 'ballnumber', 'batsman_run', 'extras_run', 'total_run', 'non_boundary', 'isWicketDelivery']
for col in num_cols_ball:
    if col in ball_df.columns:
        ball_df[col] = pd.to_numeric(ball_df[col], errors='coerce').fillna(0).astype(int)
str_cols_ball = ['batter', 'bowler', 'non-striker', 'extra_type', 'player_out', 'kind', 'fielders_involved', 'BattingTeam']
for col in str_cols_ball:
    if col in ball_df.columns:
        ball_df[col] = ball_df[col].astype(str)
match_df['ID'] = pd.to_numeric(match_df['ID'], errors='coerce').fillna(0).astype(int)
if 'Margin' in match_df.columns:
    match_df['Margin'] = pd.to_numeric(match_df['Margin'], errors='coerce').fillna(0).astype(int)
if 'Date' in match_df.columns:
    match_df['Date'] = pd.to_datetime(match_df['Date'], errors='coerce')
str_cols_match = ['City', 'Season', 'MatchNumber', 'Team1', 'Team2', 'Venue', 'TossWinner', 'TossDecision', 'SuperOver', 'WinningTeam', 'WonBy', 'Player_of_Match', 'Umpire1', 'Umpire2']
for col in str_cols_match:
    if col in match_df.columns:
        match_df[col] = match_df[col].astype(str).replace('nan', 'Unknown')
if 'ID' in ball_df.columns:
    dups_ball = ball_df.duplicated(subset=['ID', 'innings', 'overs', 'ballnumber'], keep=False)
    if dups_ball.any():
        print(f"\nFound {dups_ball.sum()} duplicate deliveries")
        ball_df = ball_df.drop_duplicates(subset=['ID', 'innings', 'overs', 'ballnumber'], keep='first')
if 'ID' in match_df.columns:
    dups_match = match_df.duplicated(subset=['ID'], keep=False)
    if dups_match.any():
        print(f"Found {dups_match.sum()} duplicate match IDs")
        match_df = match_df.drop_duplicates(subset=['ID'], keep='first')
if 'overs' in ball_df.columns:
    ball_df = ball_df[(ball_df['overs'] >= 0) & (ball_df['overs'] <= 19)]
if 'innings' in ball_df.columns:
    ball_df = ball_df[ball_df['innings'].isin([1, 2])]
try:
    ball_df.to_csv('cleaned_ball_by_ball.csv', index=False)
    match_df.to_csv('cleaned_match_level.csv', index=False)
    print("\nCleaned datasets saved in:", os.getcwd())
    print("- Ball: 'cleaned_ball_by_ball.csv'")
    print("- Match: 'cleaned_match_level.csv'")
except Exception as e:
    print(f"Error saving files: {e}")
print("\nPost-Cleaning Stats:")
print("Ball Missing Values:", ball_df.isnull().sum().sum())
if 'innings' in ball_df.columns:
    print("Ball Unique Innings:", ball_df['innings'].unique())
if 'overs' in ball_df.columns:
    print("Ball Over Range:", ball_df['overs'].min(), "-", ball_df['overs'].max())
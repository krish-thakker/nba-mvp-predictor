# main.py

import pandas as pd
from nba_api.stats.endpoints import LeagueDashPlayerStats, LeagueDashTeamStats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
import warnings
import time
import unicodedata

# Silence undefined‐metric warnings
warnings.filterwarnings("ignore", category=UserWarning)

# API retry settings
API_TIMEOUT = 60  # seconds
RETRY_DELAY = 5   # seconds between retries
MAX_RETRIES = 3   # number of retry attempts

def normalize_name(name: str) -> str:
    """
    Normalize player names by removing accents and converting to lowercase.
    """
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_name = nfkd.encode('ASCII', 'ignore').decode('utf-8')
    return ascii_name.lower()

def fetch_with_retry(fetch_fn, season: str) -> pd.DataFrame:
    """
    Helper to retry a fetch function with exponential backoff on timeout.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch_fn(season)
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"Error fetching {season} (attempt {attempt}): {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise

def fetch_season_stats(season: str) -> pd.DataFrame:
    """
    Fetch per-game stats for all players in a given season, including Plus/Minus.
    """
    stats = LeagueDashPlayerStats(
        season=season,
        season_type_all_star='Regular Season',
        per_mode_detailed='PerGame',
        measure_type_detailed_defense='Base',
        timeout=API_TIMEOUT
    )
    df = stats.get_data_frames()[0]
    # Pull PLUS_MINUS in addition to the other columns
    df = df[['PLAYER_NAME', 'TEAM_ID', 'MIN', 'PTS', 'AST', 'REB',
             'FG_PCT', 'FG3_PCT', 'FT_PCT', 'TOV', 'STL', 'BLK',
             'GP', 'PLUS_MINUS']]
    df['Season'] = season
    return df

def fetch_team_wins(season: str) -> pd.DataFrame:
    """
    Fetch team win totals for a given season using LeagueDashTeamStats.
    """
    stats = LeagueDashTeamStats(
        season=season,
        season_type_all_star='Regular Season',
        per_mode_detailed='Totals',
        measure_type_detailed_defense='Base',
        timeout=API_TIMEOUT
    )
    df = stats.get_data_frames()[0]
    df = df[['TEAM_ID', 'W']].rename(columns={'W': 'Team_Wins'})
    df['Season'] = season
    return df

def main():
    seasons = [
        '2009-10','2010-11','2011-12','2012-13','2013-14',
        '2014-15','2015-16','2016-17','2017-18','2018-19',
        '2019-20','2020-21','2021-22','2022-23','2023-24'
    ]

    # 1) Fetch player & team data with retries
    player_dfs = [fetch_with_retry(fetch_season_stats, s) for s in seasons]
    team_dfs   = [fetch_with_retry(fetch_team_wins,   s) for s in seasons]

    all_players = pd.concat(player_dfs, ignore_index=True)
    all_teams   = pd.concat(team_dfs,   ignore_index=True)

    # 2) Merge player + team, filter out <65 GP
    df = all_players.merge(all_teams, on=['Season','TEAM_ID'], how='left')
    df = df[df['GP'] >= 65]

    # 3) MVP mapping
    mvp_map = {
        '2009-10':'LeBron James','2010-11':'Derrick Rose',
        '2011-12':'LeBron James','2012-13':'LeBron James',
        '2013-14':'Kevin Durant','2014-15':'Stephen Curry',
        '2015-16':'Stephen Curry','2016-17':'Russell Westbrook',
        '2017-18':'James Harden','2018-19':'Giannis Antetokounmpo',
        '2019-20':'Giannis Antetokounmpo','2020-21':'Nikola Jokic',
        '2021-22':'Nikola Jokic','2022-23':'Joel Embiid',
        '2023-24':'Nikola Jokic'
    }
    labels = pd.DataFrame([{'Season':s,'MVP_Player':p} for s,p in mvp_map.items()])

    # 4) Merge labels & create target
    df = df.merge(labels, on='Season', how='left')
    df['is_mvp'] = (df['PLAYER_NAME'] == df['MVP_Player']).astype(int)

    # 5) Define features (now including PLUS_MINUS)
    feature_cols = [
        'MIN','PTS','AST','REB','FG_PCT','FG3_PCT',
        'FT_PCT','TOV','STL','BLK','GP','PLUS_MINUS','Team_Wins'
    ]
    X, y = df[feature_cols], df['is_mvp']

    # 6) Train/Test split (hold out 2023-24)
    train_df = df[df['Season']!='2023-24']
    test_df  = df[df['Season']=='2023-24']
    X_train, y_train = train_df[feature_cols], train_df['is_mvp']
    X_test,  y_test  = test_df[feature_cols],  test_df['is_mvp']

    # 7) Train the Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        class_weight='balanced', random_state=42
    )
    rf.fit(X_train, y_train)

    # 8) Evaluate on 2023-24
    print("=== Classification Report for 2023-24 ===")
    print(classification_report(y_test, rf.predict(X_test), zero_division=0))

    # 9) CV Macro-F1
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X, y, cv=cv, scoring='f1_macro')
    print(f"\nCV Macro-F1: {scores.mean():.3f} (scores: {scores})")

    # 10) Season-level MVP picks
    print("\n=== MVP Picks ===")
    correct = 0
    for season in seasons:
        tr = df[df['Season']!=season]
        te = df[df['Season']==season]
        rf.fit(tr[feature_cols], tr['is_mvp'])
        proba = rf.predict_proba(te[feature_cols])[:,1]
        idx = proba.argmax()
        pick = te.iloc[idx]['PLAYER_NAME']
        actual = mvp_map[season]
        mark = '✓' if normalize_name(pick)==normalize_name(actual) else '✗'
        if mark=='✓': correct += 1
        print(f"{season}: pick={pick:<25} actual={actual:<25} {mark}")
    print(f"Accuracy: {correct}/{len(seasons)}")

    # 11) Feature importances
    print("\nGini importances:\n",
          pd.Series(rf.feature_importances_, index=feature_cols)
            .sort_values(ascending=False))
    perm = permutation_importance(rf, X_test, y_test,
                                  n_repeats=30, random_state=42, n_jobs=-1)
    print("\nPermutation importances:\n",
          pd.Series(perm.importances_mean, index=feature_cols)
            .sort_values(ascending=False))

if __name__ == '__main__':
    main()

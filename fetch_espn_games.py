"""
Fetch NFL games for a given season/week range from ESPN's public JSON API
and write a `Games.csv` file suitable for use by `NFLStats.py`.

Usage:
    python fetch_espn_games.py --season 2025 --start-week 1 --end-week 18

Output columns: Date,Week,AwayTeam,HomeTeam,AwayScore,HomeScore,Winner

Requires: requests
"""
import requests
import csv
import argparse
from datetime import datetime

API_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

parser = argparse.ArgumentParser(description='Fetch NFL games from ESPN JSON API or fallback to PFR')
parser.add_argument('--season', type=int, default=2025)
parser.add_argument('--start-week', type=int, default=1)
parser.add_argument('--end-week', type=int, default=18)
parser.add_argument('--out', type=str, default='Games.csv')
parser.add_argument('--source', type=str, choices=['espn','pfr','auto'], default='auto',
                    help='Preferred source: espn (JSON API) or pfr (Pro-Football-Reference) or auto')
args = parser.parse_args()

rows = []
# ESPN request headers and retry settings to avoid simple 500s
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
}
MAX_RETRIES = 3
def clean_team_name(name):
    if not isinstance(name, str):
        return name
    # remove annotations like '*' and whitespace
    return name.replace('*','').replace('\u00a0',' ').strip()

def try_fetch_espn():
    collected = []
    for week in range(args.start_week, args.end_week + 1):
        params = {'week': week, 'season': args.season}
        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.get(API_URL, params=params, headers=HEADERS, timeout=20)
                r.raise_for_status()
                data = r.json()
                success = True
                break
            except Exception as e:
                print(f"Failed to fetch week {week} (attempt {attempt}): {e}")
        if not success:
            print(f"ESPN: skipping week {week} after {MAX_RETRIES} attempts")
            continue

        events = data.get('events', [])
        for ev in events:
            try:
                comps = ev.get('competitions', [])
                if not comps:
                    continue
                comp = comps[0]
                date_iso = comp.get('date') or ev.get('date')
                date = ''
                if date_iso:
                    try:
                        date = datetime.fromisoformat(date_iso.replace('Z', '+00:00')).strftime('%Y-%m-%d')
                    except Exception:
                        date = date_iso
                competitors = comp.get('competitors', [])
                if len(competitors) < 2:
                    continue
                home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
                away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
                if home is None or away is None:
                    away = competitors[0]
                    home = competitors[1]
                home_team = clean_team_name(home.get('team', {}).get('displayName') or home.get('team', {}).get('shortDisplayName') or home.get('team', {}).get('abbreviation'))
                away_team = clean_team_name(away.get('team', {}).get('displayName') or away.get('team', {}).get('shortDisplayName') or away.get('team', {}).get('abbreviation'))
                home_score = home.get('score')
                away_score = away.get('score')
                try:
                    home_score = int(home_score) if home_score != '' and home_score is not None else ''
                except Exception:
                    home_score = ''
                try:
                    away_score = int(away_score) if away_score != '' and away_score is not None else ''
                except Exception:
                    away_score = ''
                winner = ''
                if home_score != '' and away_score != '':
                    winner = home_team if home_score > away_score else away_team
                collected.append({'Date': date, 'Week': week, 'AwayTeam': away_team, 'HomeTeam': home_team, 'AwayScore': away_score, 'HomeScore': home_score, 'Winner': winner})
            except Exception as e:
                print(f"Error parsing event: {e}")
    return collected

rows = []
if args.source in ('espn','auto'):
    rows = try_fetch_espn()

# write CSV
with open(args.out, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['Date','Week','AwayTeam','HomeTeam','AwayScore','HomeScore','Winner'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Wrote {len(rows)} rows to {args.out}")

if len(rows) == 0:
    # ESPN failed for all weeks; fallback to Pro-Football-Reference
    try:
        import pandas as pd
        pfr_url = f"https://www.pro-football-reference.com/years/{args.season}/games.htm"
        print(f"Falling back to Pro-Football-Reference: {pfr_url}")
        # read_html can be slow; wrap in try
        tables = pd.read_html(pfr_url)
        games_table = None
        for t in tables:
            cols = [str(c) for c in t.columns]
            if any('Week' in c or 'Wk' in c for c in cols) and any('Home' in c or 'Visitor' in c or 'Away' in c for c in cols):
                games_table = t
                break
        if games_table is None:
            print("Could not find games table on PFR page.")
        else:
            dfp = games_table.copy()
            dfp.columns = [str(c) for c in dfp.columns]
            # find likely team columns
            left_col = None
            right_col = None
            for name in ['Visitor/Neutral','Visitor','Away','Team']:
                if name in dfp.columns:
                    left_col = name
                    break
            for name in ['Home/Neutral','Home']:
                if name in dfp.columns:
                    right_col = name
                    break
            # score columns often include 'Pts' or 'Score'
            score_cols = [c for c in dfp.columns if 'Pts' in c or 'Score' in c or c.strip().isdigit()]

            for _, r in dfp.iterrows():
                try:
                    # skip separator rows
                    if str(r.get('Week','')).strip().lower().startswith('playoffs'):
                        continue
                    date = r.get('Date','')
                    week = r.get('Week','')
                    away_team = clean_team_name(r.get(left_col, '')) if left_col else ''
                    home_team = clean_team_name(r.get(right_col, '')) if right_col else ''
                    away_score = ''
                    home_score = ''
                    # attempt to extract scores by searching numeric columns adjacent to teams
                    for c in dfp.columns:
                        val = r.get(c)
                        if isinstance(val, (int, float)):
                            if away_score == '':
                                away_score = int(val)
                            elif home_score == '':
                                home_score = int(val)
                    winner = ''
                    if away_score != '' and home_score != '':
                        winner = away_team if away_score > home_score else home_team
                    rows.append({'Date': date, 'Week': week, 'AwayTeam': away_team, 'HomeTeam': home_team, 'AwayScore': away_score, 'HomeScore': home_score, 'Winner': winner})
                except Exception:
                    continue
            if len(rows) > 0:
                with open(args.out, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['Date','Week','AwayTeam','HomeTeam','AwayScore','HomeScore','Winner'])
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(r)
                print(f"Wrote {len(rows)} rows to {args.out} (from PFR)")
    except Exception as e:
        print(f"PFR fallback failed: {e}")

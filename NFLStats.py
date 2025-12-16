# pip install pandas numpy scikit-learn tensorflow
# pip install xgboost

"""
NFL win-prediction script

This script trains three models (Logistic Regression, Neural Network, XGBoost)
on aggregated team statistics (offense, defense, SOS, turnovers) plus
domain features (home field advantage, weather difficulty, upset factor).

It can validate predictions against a held-out test CSV and provide an
interactive (or non-interactive) two-team matchup prediction.
"""
import xgboost as xgb
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from keras import layers, models, callbacks
import argparse
import sys
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# Attempt to import XGBoost classifier; if unavailable we fall back
# to training only Logistic Regression and the Neural Network.

# Command-line arguments
parser = argparse.ArgumentParser(description='NFL win-prediction script')
parser.add_argument('--no-interactive', action='store_true', help='Skip the two-team interactive matchup prompt')
parser.add_argument('--team1', type=str, help='First team name for non-interactive matchup')
parser.add_argument('--team2', type=str, help='Second team name for non-interactive matchup')
args = parser.parse_args()


# ------------------------------
# Load dataset CSVs
# Each CSV contains per-team aggregated statistics used as features.
# Ensure these files exist in the working directory.
# ------------------------------
Offense = pd.read_csv("Offense.csv")
Defense = pd.read_csv("Defense.csv")
SOS = pd.read_csv("SOS.csv")
Turnovers = pd.read_csv("Turnovers.csv")
Record = pd.read_csv("WinsLosses.csv")
HomeFieldAdv = pd.read_csv("HomeFieldAdv.csv")
WeatherEffects = pd.read_csv("WeatherEffects.csv")
UpsetFactor = pd.read_csv("UpsetFactor.csv")

# Normalize team name strings
# We strip whitespace to avoid merge mismatches when joining on the 'Team' column.
Offense['Team'] = Offense['Team'].str.strip()
Defense['Team'] = Defense['Team'].str.strip()
SOS['Team'] = SOS['Team'].str.strip()
Turnovers['Team'] = Turnovers['Team'].str.strip()
Record['Team'] = Record['Team'].str.strip()
HomeFieldAdv['Team'] = HomeFieldAdv['Team'].str.strip()
WeatherEffects['Team'] = WeatherEffects['Team'].str.strip()
UpsetFactor['Team'] = UpsetFactor['Team'].str.strip()

# Map common short team names (appearing in some CSVs) to the canonical
# full team names used across the other CSVs. This ensures consistent keys
# for merges (e.g., 'Eagles' -> 'Philadelphia Eagles').
team_mapping = {
    'Eagles': 'Philadelphia Eagles',
    'Cowboys': 'Dallas Cowboys',
    'Commanders': 'Washington Commanders',
    'Giants': 'New York Giants',
    'Buccaneers': 'Tampa Bay Buccaneers',
    'Panthers': 'Carolina Panthers',
    'Falcons': 'Atlanta Falcons',
    'Saints': 'New Orleans Saints',
    'Rams': 'Los Angeles Rams',
    'Seahawks': 'Seattle Seahawks',
    '49ers': 'San Francisco 49ers',
    'Cardinals': 'Arizona Cardinals',
    'Packers': 'Green Bay Packers',
    'Bears': 'Chicago Bears',
    'Lions': 'Detroit Lions',
    'Vikings': 'Minnesota Vikings',
    'Broncos': 'Denver Broncos',
    'Chargers': 'Los Angeles Chargers',
    'Chiefs': 'Kansas City Chiefs',
    'Raiders': 'Las Vegas Raiders',
    'Patriots': 'New England Patriots',
    'Bills': 'Buffalo Bills',
    'Dolphins': 'Miami Dolphins',
    'Jets': 'New York Jets',
    'Jaguars': 'Jacksonville Jaguars',
    'Texans': 'Houston Texans',
    'Colts': 'Indianapolis Colts',
    'Titans': 'Tennessee Titans',
    'Steelers': 'Pittsburgh Steelers',
    'Ravens': 'Baltimore Ravens',
    'Bengals': 'Cincinnati Bengals',
    'Browns': 'Cleveland Browns'
}

# Apply the mapping to the Record dataframe, and strip any extra
# whitespace from column headers to avoid accidental mismatches.
Record['Team'] = Record['Team'].map(team_mapping).fillna(Record['Team'])
Record.columns = Record.columns.str.strip()

# Strip whitespace from all column names in the team CSVs as well
Offense.columns = Offense.columns.str.strip()
Defense.columns = Defense.columns.str.strip()
SOS.columns = SOS.columns.str.strip()
Turnovers.columns = Turnovers.columns.str.strip()

# Build a single dataframe `df` with all team-level features by merging
# offense, defense, SOS, turnovers, and the records. We keep left joins
# for auxiliary datasets so missing values don't drop teams unintentionally.
df = Offense.merge(Defense, on="Team", suffixes=('_off', '_def'))
df = df.merge(SOS, on="Team")
df = df.merge(Turnovers, on="Team")
df = df.merge(Record, on="Team")

# Merge home field advantage (take Rank as feature) and rename column
df = df.merge(HomeFieldAdv[['Team', 'Rank']], on='Team', how='left')
df.rename(columns={'Rank': 'HomeFieldAdv_Rank'}, inplace=True)

# Merge weather effects (difficulty rating)
df = df.merge(WeatherEffects[['Team', 'Weather_Difficulty_Rating']], on='Team', how='left')

# Merge upset factor (volatility / upset likelihood)
df = df.merge(UpsetFactor[['Team', 'Upset_Factor']], on='Team', how='left')

# Create target column for supervised learning: 1 if team had a winning
# record (Wins > Losses), otherwise 0. You can replace this with game-
# level targets later if desired.
df['target'] = (df['Wins'] > df['Losses']).astype(int)

# Prepare feature matrix `x` and label vector `y`.
# We drop identifying/non-numeric columns that shouldn't be used as features.
x = df.drop(columns=["Team", "Wins", "Losses", "OCR", "target", "Stadium"], errors='ignore')
y = df['target']  # target label (1 = winning record)

# Debug / sanity checks: print dtypes and check missing values so we can
# catch problems early during development.
print("Data types:")
print(x.dtypes)
print("\nDataframe info:")
print(x.info())

# Check for NaN values in the merged dataframe and report them.
if df.isna().any().any():
    print("\nWARNING: NaN values found in dataframe!")
    print(df.isna().sum())
    print("\nTeams with NaN values:")
    print(df[df.isna().any(axis=1)][['Team', 'Wins', 'Losses']])
else:
    print("\nNo NaN values found. Data looks good!")

# Split dataset into train/test and standardize features. We fit the
# StandardScaler on the training split and reuse it to transform test
# and later prediction inputs so there is no data leakage.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42) # 75% train, 25% test
scaler = StandardScaler() # Standardize features
x_train = scaler.fit_transform(x_train) # Fit on training data
x_test = scaler.transform(x_test) # Transform test data
le = LabelEncoder() # Encode target labels
y_train = le.fit_transform(y_train) # Fit and transform training labels
y_test = le.transform(y_test) # Transform test labels

# --------------------------
# Logistic Regression
# A simple baseline linear classifier for binary win/loss prediction.
# We use it as a baseline to compare against the NN and XGBoost models.
# --------------------------
log_reg = LogisticRegression(max_iter=1000) # Increase max_iter
log_reg.fit(x_train, y_train) # Train model
y_pred_log_reg = log_reg.predict(x_test) # Predict on test data
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg) # Confusion matrix
print("Logistic Regression Confusion Matrix:")
print(cm_log_reg)
# --------------------------
# XGBoost (gradient-boosted trees)
# We train XGBoost if the package is available. It often provides strong
# tabular-data performance; we catch training errors and continue if it fails.
# --------------------------
if XGBClassifier is not None:
    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, max_depth=4, random_state=42, verbosity=0)
    try:
        # train without early stopping for compatibility
        xgb_clf.fit(x_train, y_train)
    except Exception as e:
        print(f"XGBoost training error: {e}")
        xgb_clf = None

    if xgb_clf is not None:
        y_pred_xgb = xgb_clf.predict(x_test)
        cm_xgb = confusion_matrix(y_test, y_pred_xgb)
        print("XGBoost Confusion Matrix:")
        print(cm_xgb)
else:
    print("XGBoost not installed; skipping XGBoost model.")
# --------------------------
# Neural Network (Keras)
# A small fully-connected network trained on the same features. We use
# early stopping on validation loss to prevent overfitting.
# --------------------------
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))  # Input layer
model.add(layers.Dense(32, activation='relu'))  # Hidden layer
model.add(layers.Dense(1, activation='sigmoid'))  # Output layer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile model
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Early stopping
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.15, callbacks=[early_stopping], verbose=0)  # Train model
y_pred_nn = (model.predict(x_test, verbose=0) > 0.5).astype("int32")  # Predict on test data
cm_nn = confusion_matrix(y_test, y_pred_nn)  # Confusion matrix
print("Neural Network Confusion Matrix:")
print(cm_nn)

# ===== GAME PREDICTION =====
print("\n" + "="*60)
print("PREDICTING GAMES WITH UNSEEN DATA")
print("="*60)

# --------------------------
# Load and prepare unseen/test games for prediction
# The script expects per-team test CSVs mirroring the training features.
# We standardize, map team names, then merge the same way as the training pipeline.
# --------------------------

# Load test data (you need to create these CSV files with game stats)
try:
    OffenseTest = pd.read_csv("OffenseTest.csv")
    DefenseTest = pd.read_csv("DefenseTest.csv")
    SOSTest = pd.read_csv("SOSTest.csv")
    TurnoversTest = pd.read_csv("TurnoversTest.csv")
    HomeFieldAdvTest = pd.read_csv("HomeFieldAdvTest.csv")
    WeatherEffectsTest = pd.read_csv("WeatherEffectsTest.csv")
    UpsetFactorTest = pd.read_csv("UpsetFactorTest.csv")
    
    # Standardize test data team names
    OffenseTest['Team'] = OffenseTest['Team'].str.strip()
    DefenseTest['Team'] = DefenseTest['Team'].str.strip()
    SOSTest['Team'] = SOSTest['Team'].str.strip()
    TurnoversTest['Team'] = TurnoversTest['Team'].str.strip()
    HomeFieldAdvTest['Team'] = HomeFieldAdvTest['Team'].str.strip()
    WeatherEffectsTest['Team'] = WeatherEffectsTest['Team'].str.strip()
    UpsetFactorTest['Team'] = UpsetFactorTest['Team'].str.strip()
    
    # Strip whitespace from test data column names (headers may vary)
    OffenseTest.columns = OffenseTest.columns.str.strip()
    DefenseTest.columns = DefenseTest.columns.str.strip()
    SOSTest.columns = SOSTest.columns.str.strip()
    TurnoversTest.columns = TurnoversTest.columns.str.strip()
    HomeFieldAdvTest.columns = HomeFieldAdvTest.columns.str.strip()
    WeatherEffectsTest.columns = WeatherEffectsTest.columns.str.strip()
    UpsetFactorTest.columns = UpsetFactorTest.columns.str.strip()

    # DEBUG: print shapes and columns to diagnose merge issues
    print("DEBUG: OffenseTest.shape:", OffenseTest.shape)
    print("DEBUG: OffenseTest.columns:", OffenseTest.columns.tolist())
    print("DEBUG: DefenseTest.shape:", DefenseTest.shape)
    print("DEBUG: DefenseTest.columns:", DefenseTest.columns.tolist())
    print("DEBUG: SOSTest.shape:", SOSTest.shape)
    print("DEBUG: SOSTest.columns:", SOSTest.columns.tolist())
    print("DEBUG: TurnoversTest.shape:", TurnoversTest.shape)
    print("DEBUG: TurnoversTest.columns:", TurnoversTest.columns.tolist())
    print("DEBUG: HomeFieldAdvTest.shape:", HomeFieldAdvTest.shape)
    print("DEBUG: HomeFieldAdvTest.columns:", HomeFieldAdvTest.columns.tolist())
    print("DEBUG: WeatherEffectsTest.shape:", WeatherEffectsTest.shape)
    print("DEBUG: WeatherEffectsTest.columns:", WeatherEffectsTest.columns.tolist())
    print("DEBUG: UpsetFactorTest.shape:", UpsetFactorTest.shape)
    print("DEBUG: UpsetFactorTest.columns:", UpsetFactorTest.columns.tolist())
    
    # Map short names to canonical full names where applicable. Use
    # fillna(...) to preserve already-correct full names.
    OffenseTest['Team'] = OffenseTest['Team'].map(team_mapping).fillna(OffenseTest['Team'])
    DefenseTest['Team'] = DefenseTest['Team'].map(team_mapping).fillna(DefenseTest['Team'])
    SOSTest['Team'] = SOSTest['Team'].map(team_mapping).fillna(SOSTest['Team'])
    TurnoversTest['Team'] = TurnoversTest['Team'].map(team_mapping).fillna(TurnoversTest['Team'])
    HomeFieldAdvTest['Team'] = HomeFieldAdvTest['Team'].map(team_mapping).fillna(HomeFieldAdvTest['Team'])
    WeatherEffectsTest['Team'] = WeatherEffectsTest['Team'].map(team_mapping).fillna(WeatherEffectsTest['Team'])
    UpsetFactorTest['Team'] = UpsetFactorTest['Team'].map(team_mapping).fillna(UpsetFactorTest['Team'])
    
    # Merge test data the same way as training data (keep column names aligned)
    test_df = OffenseTest.merge(DefenseTest, on="Team", suffixes=('_off', '_def'))
    test_df = test_df.merge(SOSTest, on="Team")
    test_df = test_df.merge(TurnoversTest, on="Team")
    # Rename Rank column in HomeFieldAdvTest before merging to avoid conflicts
    homeadv_cols = HomeFieldAdvTest[['Team', 'Rank']].copy()
    homeadv_cols.rename(columns={'Rank': 'HomeFieldAdv_Rank'}, inplace=True)
    # DEBUG: inspect test_df before mapping
    print("DEBUG BEFORE MAPPING: test_df.shape=", test_df.shape)
    print("DEBUG BEFORE MAPPING: test_df unique teams=", test_df['Team'].nunique() if 'Team' in test_df.columns else 'NO TEAM COL')

    # Use dictionary mapping instead of DataFrame.merge to avoid
    # potential Cartesian-product or duplicate-column merges that can
    # consume huge amounts of memory. Mapping is safer for one-to-one
    # lookups like Team -> Rank.
    homeadv_map = dict(HomeFieldAdvTest[['Team', 'Rank']].values)
    test_df['HomeFieldAdv_Rank'] = test_df['Team'].map(homeadv_map)

    weather_map = dict(WeatherEffectsTest[['Team', 'Weather_Difficulty_Rating']].values)
    test_df['Weather_Difficulty_Rating'] = test_df['Team'].map(weather_map)

    upset_map = dict(UpsetFactorTest[['Team', 'Upset_Factor']].values)
    test_df['Upset_Factor'] = test_df['Team'].map(upset_map)
    
    # Prepare features for prediction (same columns as training, in same order)
    x_predict = test_df.drop(columns=["Team", "OCR", "Stadium"], errors='ignore')
    
    # Ensure test data has the same columns in the same order as training
    x_predict = x_predict[x.columns]
    
    # Scale using the same scaler from training
    x_predict_scaled = scaler.transform(x_predict)
    
    # Make predictions with both models
    log_reg_predictions = log_reg.predict(x_predict_scaled)
    nn_predictions = (model.predict(x_predict_scaled, verbose=0) > 0.5).astype("int32").flatten()
    # XGBoost predictions (if available)
    if XGBClassifier is not None:
        xgb_predictions = xgb_clf.predict(x_predict_scaled)
        test_df['pred_xgb'] = xgb_predictions
    else:
        xgb_predictions = None

    # attach predictions to dataframe for easy validation and downstream display
    test_df['pred_log'] = log_reg_predictions
    test_df['pred_nn'] = nn_predictions

    # If a Wins/Losses test file exists, merge it to compute real outcomes and validate
    try:
        WinsTest = pd.read_csv("WinsLossesTest.csv")
        WinsTest['Team'] = WinsTest['Team'].str.strip()
        WinsTest.columns = WinsTest.columns.str.strip()
        # Map short names to full names (same mapping used earlier)
        WinsTest['Team'] = WinsTest['Team'].map(team_mapping).fillna(WinsTest['Team'])

        # Merge into test_df to get actual results
        test_df = test_df.merge(WinsTest, on='Team', how='left')
        # Compute actual label
        test_df['actual'] = (test_df['Wins'] > test_df['Losses']).astype(int)

        # Compare predictions to actuals and report accuracy / confusion matrices
        from sklearn.metrics import accuracy_score
        log_acc = accuracy_score(test_df['actual'], test_df['pred_log'])
        nn_acc = accuracy_score(test_df['actual'], test_df['pred_nn'])
        cm_log_test = confusion_matrix(test_df['actual'], test_df['pred_log'])
        cm_nn_test = confusion_matrix(test_df['actual'], test_df['pred_nn'])
        if XGBClassifier is not None:
            xgb_acc = accuracy_score(test_df['actual'], test_df['pred_xgb'])
            cm_xgb_test = confusion_matrix(test_df['actual'], test_df['pred_xgb'])

        # Display results with validation
        header = f"{'Team':<30} {'Actual':<8} {'LogReg':<8} {'NN':<8}"
        if XGBClassifier is not None:
            header = f"{'Team':<30} {'Actual':<8} {'LogReg':<8} {'NN':<8} {'XGBoost':<8} {'Upset %':<10}"
        print("\nGAME PREDICTIONS FOR TEST DATA (with real outcomes):")
        print("-" * (100 if XGBClassifier is None else 140))
        print(header)
        print("-" * (100 if XGBClassifier is None else 140))
        for i, row in test_df.reset_index().iterrows():
            team = row['Team']
            actual = 'WIN' if row['actual'] == 1 else 'LOSS'
            lg = 'WIN' if row['pred_log'] == 1 else 'LOSS'
            nn = 'WIN' if row['pred_nn'] == 1 else 'LOSS'
            upset_pct = f"{row['Upset_Factor']*100:.1f}%" if 'Upset_Factor' in row else "N/A"
            if XGBClassifier is not None:
                xgbp = 'WIN' if row['pred_xgb'] == 1 else 'LOSS'
                print(f"{team:<30} {actual:<8} {lg:<8} {nn:<8} {xgbp:<8} {upset_pct:<10}")
            else:
                print(f"{team:<30} {actual:<8} {lg:<8} {nn:<8} {upset_pct:<10}")
        print("-" * (100 if XGBClassifier is None else 140))
        print(f"Logistic Regression accuracy on test games: {log_acc:.4f}")
        print(f"Neural Network accuracy on test games:      {nn_acc:.4f}")
        if XGBClassifier is not None:
            print(f"XGBoost accuracy on test games:             {xgb_acc:.4f}")
        print("\nLogReg confusion matrix:\n", cm_log_test)
        print("\nNN confusion matrix:\n", cm_nn_test)
        if XGBClassifier is not None:
            print("\nXGBoost confusion matrix:\n", cm_xgb_test)

        # Print mismatches: find rows where ANY model prediction differs from actual
        cols_to_check = ['pred_log','pred_nn'] + (['pred_xgb'] if XGBClassifier is not None else [])
        # Compare each prediction column to the 'actual' Series row-wise
        mismatch_mask = test_df[cols_to_check].ne(test_df['actual'], axis=0).any(axis=1)
        mismatches = test_df.loc[mismatch_mask, ['Team','Wins','Losses','actual'] + cols_to_check]
        if not mismatches.empty:
            print('\nMismatches (teams where any model disagreed with actual outcome):')
            for _, r in mismatches.iterrows():
                parts = [f"{r['Team']}: actual={'WIN' if r['actual']==1 else 'LOSS'}"]
                parts.append(f"LogReg={'WIN' if r['pred_log']==1 else 'LOSS'}")
                parts.append(f"NN={'WIN' if r['pred_nn']==1 else 'LOSS'}")
                if XGBClassifier is not None:
                    parts.append(f"XGB={'WIN' if r['pred_xgb']==1 else 'LOSS'}")
                print(' - ' + ', '.join(parts))
    except FileNotFoundError:
        # No ground-truth file; just display predictions
        print("\nGAME PREDICTIONS FOR TEST DATA:")
        print("-" * 80)
        print(f"{'Team':<25} {'Logistic Reg':<20} {'Neural Network':<20}")
        print("-" * 80)
        for i, team in enumerate(test_df['Team']):
            log_pred = "WIN (1)" if log_reg_predictions[i] == 1 else "LOSS (0)"
            nn_pred = "WIN (1)" if nn_predictions[i] == 1 else "LOSS (0)"
            print(f"{team:<25} {log_pred:<20} {nn_pred:<20}")
        print("-" * 80)
    
except FileNotFoundError as e:
    print(f"\nTest data files not found: {e}")
    print("To use game predictions, create test CSV files:")
    print("  - OffenseTest.csv")
    print("  - DefenseTest.csv")
    print("  - SOSTest.csv")
    print("  - TurnoversTest.csv")

# ===== TWO-TEAM MATCHUP PREDICTION =====
# Interactive two-team matchup: prompts the user for two teams and prints
# each model's prediction plus a simple consensus vote. You can skip or
# supply teams non-interactively with the command-line flags.
print("\n" + "="*60)
print("PREDICT WINNER BETWEEN TWO TEAMS")
print("="*60)

# If both team args provided, use them non-interactively
if args.team1 and args.team2:
    team1 = args.team1
    team2 = args.team2

    # map partial names to full available team names if needed
    available_teams = df['Team'].unique()
    # try exact map first, otherwise do partial match
    def resolve_team(name):
        name = name.strip()
        for t in available_teams:
            if name.lower() == t.lower():
                return t
        matches = [t for t in available_teams if name.lower() in t.lower()]
        return matches[0] if len(matches) == 1 else None

    t1 = resolve_team(team1)
    t2 = resolve_team(team2)
    if t1 is None or t2 is None:
        print("Could not resolve team names from --team1/--team2 arguments. Use --no-interactive to skip interactive prompt.")
        sys.exit(1)
    team1 = t1
    team2 = t2

elif args.no_interactive:
    print("\nSkipping interactive two-team matchup (--no-interactive). Exiting.")
    sys.exit(0)

# Get available teams from the training data
available_teams = df['Team'].unique()
print(f"\nAvailable teams ({len(available_teams)}):")
for i, team in enumerate(sorted(available_teams), 1):
    print(f"  {i}. {team}")

# If team names weren't provided via CLI, prompt interactively. If they
# were provided earlier (args.team1 && args.team2) we've already resolved
# them and can skip this prompt.
if not (args.team1 and args.team2):
    # Get user input for two teams
    while True:
        try:
            team1_input = input("\nEnter first team name (or part of it): ").strip()
            team2_input = input("Enter second team name (or part of it): ").strip()
            
            # Find matching teams (case-insensitive, partial match)
            team1_matches = [t for t in available_teams if team1_input.lower() in t.lower()]
            team2_matches = [t for t in available_teams if team2_input.lower() in t.lower()]
            
            if not team1_matches:
                print(f"No team found matching '{team1_input}'. Please try again.")
                continue
            if not team2_matches:
                print(f"No team found matching '{team2_input}'. Please try again.")
                continue
            
            if len(team1_matches) > 1:
                print(f"Multiple matches for '{team1_input}': {', '.join(team1_matches)}")
                continue
            if len(team2_matches) > 1:
                print(f"Multiple matches for '{team2_input}': {', '.join(team2_matches)}")
                continue
            
            team1 = team1_matches[0]
            team2 = team2_matches[0]
            
            if team1 == team2:
                print("Please enter two different teams.")
                continue
            
            break
        
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()

# Get team stats from training data
team1_stats = df[df['Team'] == team1].drop(columns=["Team", "Wins", "Losses", "OCR", "target"]).values[0]
team2_stats = df[df['Team'] == team2].drop(columns=["Team", "Wins", "Losses", "OCR", "target"]).values[0]

# Ensure stats are in the same order as training features
team1_stats = pd.DataFrame([team1_stats], columns=x.columns)
team2_stats = pd.DataFrame([team2_stats], columns=x.columns)

# Scale the stats
team1_scaled = scaler.transform(team1_stats)
team2_scaled = scaler.transform(team2_stats)

# Make predictions
log_reg_team1 = log_reg.predict(team1_scaled)[0]
log_reg_team2 = log_reg.predict(team2_scaled)[0]
nn_team1 = (model.predict(team1_scaled, verbose=0) > 0.5).astype("int32")[0][0]
nn_team2 = (model.predict(team2_scaled, verbose=0) > 0.5).astype("int32")[0][0]
if XGBClassifier is not None:
    xgb_team1 = xgb_clf.predict(team1_scaled)[0]
    xgb_team2 = xgb_clf.predict(team2_scaled)[0]
else:
    xgb_team1 = None
    xgb_team2 = None

# Display results
print("\n" + "="*60)
print(f"MATCHUP: {team1} vs {team2}")
print("="*60)
print(f"\n{'Model':<25} {'Team 1 ({})': <30} {'Team 2 ({})': <30}")
print(f"{'': <25} {team1: <30} {team2: <30}")
print("-" * 85)
print(f"{'Logistic Regression': <25} {'WIN' if log_reg_team1 == 1 else 'LOSS': <30} {'WIN' if log_reg_team2 == 1 else 'LOSS': <30}")
print(f"{'Neural Network': <25} {'WIN' if nn_team1 == 1 else 'LOSS': <30} {'WIN' if nn_team2 == 1 else 'LOSS': <30}")
if XGBClassifier is not None:
    print(f"{'XGBoost': <25} {'WIN' if xgb_team1 == 1 else 'LOSS': <30} {'WIN' if xgb_team2 == 1 else 'LOSS': <30}")
print("-" * 85)

# (Upset percentages will be displayed after consensus is computed)

# Determine consensus winner
log_reg_winner = team1 if log_reg_team1 > log_reg_team2 else team2
nn_winner = team1 if nn_team1 > nn_team2 else team2
if XGBClassifier is not None:
    xgb_winner = team1 if xgb_team1 > xgb_team2 else team2
else:
    xgb_winner = None

print(f"\nLogistic Regression predicts: {log_reg_winner} wins")
print(f"Neural Network predicts: {nn_winner} wins")
if XGBClassifier is not None:
    print(f"XGBoost predicts: {xgb_winner} wins")

# Compute a simple vote-based consensus across available models.
votes_team1 = int(log_reg_team1) + int(nn_team1) + (int(xgb_team1) if XGBClassifier is not None else 0)
votes_team2 = int(log_reg_team2) + int(nn_team2) + (int(xgb_team2) if XGBClassifier is not None else 0)
final_winner = team1 if votes_team1 >= votes_team2 else team2

if log_reg_winner == nn_winner:
    print(f"\nBoth models agree: {log_reg_winner} wins this matchup!")
else:
    if XGBClassifier is not None:
        consensus = team1 if votes_team1 > votes_team2 else team2
        print(f"\nModels disagree. 3-model vote: {consensus} wins ({votes_team1} vs {votes_team2})")
    else:
        print(f"\nModels disagree - it's a close call! (vote: {votes_team1} vs {votes_team2})")

# After determining the consensus winner, display the upset chance for the
# losing team only — the chance an upset could occur against the predicted winner.
loser = team2 if final_winner == team1 else team1
loser_upset_val = None
if 'Upset_Factor' in df.columns:
    s = df.loc[df['Team'] == loser, 'Upset_Factor']
    if not s.empty and pd.notna(s.values[0]):
        loser_upset_val = s.values[0]

if loser_upset_val is not None:
    # left column label, then show winner blank and loser upset percentage aligned
    if final_winner == team1:
        # team1 wins; show upset % under team2 column
        print(f"{'Chances of Upset:': <25} {'':<30} {loser_upset_val*100:.1f}%")
    else:
        print(f"{'Chances of Upset:': <25} {loser_upset_val*100:.1f}%")
    print("-" * 85)

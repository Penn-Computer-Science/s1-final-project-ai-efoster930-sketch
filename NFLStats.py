# pip install pandas numpy scikit-learn tensorflow
# pip install xgboost
# python -m pip install -r requirements.txt
"""
NFL win-prediction script

This script demonstrates a compact, end-to-end pipeline for predicting
team-level winning records (binary target) from aggregated season-level
statistics. It shows common data-preparation steps, several modelling
approaches (linear model, tree ensembles, and a small neural network),
and how to apply those models to unseen test data and to head-to-head
matchup predictions.

Key points and components:
- Input CSVs: per-team aggregated statistics such as offense/defense
    yards and points per game, strength-of-schedule (SOS), turnovers,
    home-field advantage rank, weather difficulty, and an Upset_Factor
    (domain-derived volatility metric).
- Models trained: LogisticRegression (interpretable baseline),
    DecisionTree and RandomForest (tree-based baselines), a Keras
    feed-forward Neural Network, and XGBoost (if installed).
- Evaluation: confusion matrices and accuracy on held-out test split,
    plus optional validation against `WinsLossesTest.csv` when present.
- Matchup prediction: scale and predict two teams' feature vectors,
    aggregate model votes/probabilities into a simple ensemble, and
    compute an "adjusted upset" probability that mixes model confidence
    with the dataset Upset_Factor and historical win ratio.

Usage notes:
- Run non-interactively with `--no-interactive` or specify two teams
    with `--team1` and `--team2` for immediate matchup output.
- The script is intentionally conservative with memory use when
    combining test auxiliary tables: one-to-one mappings are applied
    (Team -> Rank / Weather / Upset) instead of large merges.
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import argparse
import sys
import os
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# Attempt to import XGBoost classifier; if unavailable we fall back
# to training only Logistic Regression and the Neural Network.

# Command-line arguments
# The script supports three modes of operation:
# 1. Interactive (default): prompts user for two team names and displays matchup.
# 2. Non-interactive (--no-interactive): trains models, validates on test data,
#    then exits without requesting user input. Useful for automated pipelines.
# 3. Non-interactive with teams (--team1 + --team2): runs training and validation,
#    then immediately predicts the specified matchup without user interaction.
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

# Optional: if a game-level file exists, build a per-game dataset by
# aligning each game's teams to the team-level stats. The Games CSV
# should contain at least `HomeTeam` and `AwayTeam`, and either
# `HomeScore`/`AwayScore` or a `Winner` column with the winning team name.
use_game_level = False
games_X = None
games_y = None
if os.path.exists('Games.csv'):
    try:
        games = pd.read_csv('Games.csv')
        # normalize team name columns if present
        for c in ['HomeTeam', 'AwayTeam', 'Winner']:
            if c in games.columns:
                games[c] = games[c].astype(str).str.strip()
        # map short names to canonical
        for c in ['HomeTeam', 'AwayTeam', 'Winner']:
            if c in games.columns:
                games[c] = games[c].map(team_mapping).fillna(games[c])

        rows = []
        targets = []
        missing = 0
        for _, g in games.iterrows():
            if 'HomeTeam' not in g or 'AwayTeam' not in g:
                continue
            ht = g['HomeTeam']
            at = g['AwayTeam']
            # find team stat rows
            r_ht = df.loc[df['Team'] == ht]
            r_at = df.loc[df['Team'] == at]
            if r_ht.empty or r_at.empty:
                missing += 1
                continue
            # extract feature vectors using same columns as `x`
            v_ht = r_ht.drop(columns=["Team", "Wins", "Losses", "OCR", "target", "Stadium"], errors='ignore').iloc[0]
            v_at = r_at.drop(columns=["Team", "Wins", "Losses", "OCR", "target", "Stadium"], errors='ignore').iloc[0]
            # feature: home minus away (keeps directionality for home advantage)
            diff = (v_ht - v_at).astype(float)
            # also include a binary home indicator (always 1 for home row)
            diff['Is_Home'] = 1.0
            rows.append(diff.values)

            # determine target: 1 if home team won, 0 otherwise
            winner = None
            if 'HomeScore' in games.columns and 'AwayScore' in games.columns:
                try:
                    hs = float(g['HomeScore'])
                    as_ = float(g['AwayScore'])
                    winner = ht if hs > as_ else at
                except Exception:
                    winner = None
            if winner is None and 'Winner' in games.columns:
                winner = g['Winner']
            if winner is None:
                # cannot determine winner; skip
                missing += 1
                rows.pop()
                continue
            targets.append(1 if winner == ht else 0)

        if len(rows) > 0:
            import numpy as _np
            games_X = _np.vstack(rows)
            games_y = _np.array(targets, dtype=int)
            use_game_level = True
            print(f"Loaded Games.csv: {len(rows)} games (skipped {missing} with missing teams/scores)")
        else:
            print("Games.csv found but no usable rows after alignment; falling back to season-level training.")
    except Exception as e:
        print(f"Error loading Games.csv: {e}; falling back to season-level training.")

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

# Split dataset into train/test and standardize features.
# If `Games.csv` provided and aligned, train on game-level features; else use
# season-level team features (the original behavior).
if use_game_level and games_X is not None and games_y is not None:
    # games_X already contains numeric numpy data; split and scale
    x_train, x_test, y_train, y_test = train_test_split(
        games_X, games_y, test_size=0.25, random_state=42
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
else:
    # season-level features (team-level)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )  # 75% train, 25% test
    scaler = StandardScaler()  # Standardize features (zero mean, unit variance)
    x_train = scaler.fit_transform(x_train)  # Fit on training data only
    x_test = scaler.transform(x_test)  # Apply same transform to test data

# Encode labels when needed
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# --------------------------
# Logistic Regression (interpretable baseline)
# - Linear model that learns coefficients for each standardized feature.
# - Fast to train and provides a useful baseline; coefficients can be
#   inspected to see which features move predictions in which direction.
# - We use `predict_proba` later when constructing ensemble probabilities.
# --------------------------
log_reg = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
log_reg.fit(x_train, y_train)  # Train model
y_pred_log_reg = log_reg.predict(x_test)  # Predict on test data
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)  # Confusion matrix
print("Logistic Regression Confusion Matrix:")
print(cm_log_reg)
# --------------------------
# XGBoost (gradient-boosted trees)
# We train XGBoost if the package is available. It often provides strong
# tabular-data performance; we catch training errors and continue if it fails.
# --------------------------
if XGBClassifier is not None:
    # XGBoost hyperparameters chosen conservatively for small datasets.
    # - `n_estimators` controls number of trees, `max_depth` limits tree size.
    # - We silence verbose logging for cleaner runtime output.
    xgb_clf = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=200,
        max_depth=4,
        random_state=42,
        verbosity=0,
    )
    try:
        # Train; XGBoost is robust on tabular data but can be sensitive to
        # mislabeled or heavily imbalanced targets. If it fails we continue
        # gracefully with the remaining models.
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
# Decision Tree & Random Forest
# - Tree models add non-linear interactions and different inductive biases
#   to the ensemble. Random Forest aggregates many trees to reduce variance.
# - We include them to diversify the ensemble of model opinions.
# --------------------------
dt_clf = None
rf_clf = None
try:
    dt_clf = DecisionTreeClassifier(random_state=42, max_depth=6)
    dt_clf.fit(x_train, y_train)
    y_pred_dt = dt_clf.predict(x_test)
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    print("Decision Tree Confusion Matrix:")
    print(cm_dt)
except Exception as e:
    print("Decision Tree training error:", e)
    dt_clf = None

try:
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf_clf.fit(x_train, y_train)
    y_pred_rf = rf_clf.predict(x_test)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print("Random Forest Confusion Matrix:")
    print(cm_rf)
except Exception as e:
    print("Random Forest training error:", e)
    rf_clf = None
# --------------------------
# Neural Network (Keras)
# - A compact feed-forward network with two hidden layers. The final
#   sigmoid output is interpreted as the probability of the positive
#   class (winning record). We use `predict` to obtain probabilities and
#   threshold at 0.5 for class predictions.
# - EarlyStopping on a small validation split prevents overfitting when
#   training for many epochs; this is important with limited data.
# --------------------------
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))  # Input layer
model.add(layers.Dense(32, activation='relu'))  # Hidden layer
model.add(layers.Dense(1, activation='sigmoid'))  # Output layer (probability)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile model
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Early stopping
model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.15,
    callbacks=[early_stopping],
    verbose=0,
)  # Train model
y_pred_nn = (model.predict(x_test, verbose=0) > 0.5).astype("int32")  # Predict on test data (binary)
cm_nn = confusion_matrix(y_test, y_pred_nn)  # Confusion matrix
print("Neural Network Confusion Matrix:")
print(cm_nn)

# ===== GAME PREDICTION =====
print("\n" + "="*60)
print("PREDICTING GAMES WITH UNSEEN DATA")
print("="*60)

# --------------------------
# Load and prepare unseen/test games for prediction
#
# The pipeline expects a collection of per-team CSVs that mirror the
# columns used during training (offense/defense/SOS/Turnovers plus the
# optional auxiliary tables). Test CSVs typically represent the upcoming
# week's team-level stats or an alternate season snapshot. Important
# considerations:
# - Column headers may vary slightly between files (extra whitespace,
#   year suffixes, or abbreviated team names). We defensively strip
#   whitespace and map common short names to canonical full names.
# - When joining auxiliary tables (home-field rank, weather, upset
#   factor), we prefer dictionary-based one-to-one mapping (Team -> val)
#   instead of large DataFrame merges to avoid accidental Cartesian
#   products and excessive memory use.
# - If ground-truth `WinsLossesTest.csv` is present we validate
#   predictions against actual outcomes and report mismatches.
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

    # Decision Tree and Random Forest predictions (if available)
    if 'dt_clf' in globals() and dt_clf is not None:
        dt_predictions = dt_clf.predict(x_predict_scaled)
        test_df['pred_dt'] = dt_predictions
    else:
        dt_predictions = None

    if 'rf_clf' in globals() and rf_clf is not None:
        rf_predictions = rf_clf.predict(x_predict_scaled)
        test_df['pred_rf'] = rf_predictions
    else:
        rf_predictions = None

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

        # Compare predictions to actuals and report accuracy / confusion matrices.
        # This validation step shows how well each model performs on the test data
        # when ground-truth outcomes are known. Accuracy gives a simple percentage;
        # confusion matrices show false positives / false negatives, revealing if
        # a model is systematically biased toward predicting wins or losses.
        from sklearn.metrics import accuracy_score
        log_acc = accuracy_score(test_df['actual'], test_df['pred_log'])
        nn_acc = accuracy_score(test_df['actual'], test_df['pred_nn'])
        cm_log_test = confusion_matrix(test_df['actual'], test_df['pred_log'])
        cm_nn_test = confusion_matrix(test_df['actual'], test_df['pred_nn'])
        if XGBClassifier is not None:
            xgb_acc = accuracy_score(test_df['actual'], test_df['pred_xgb'])
            cm_xgb_test = confusion_matrix(test_df['actual'], test_df['pred_xgb'])

        # Display results with validation
        # Build header dynamically depending on which model prediction columns exist
        cols_display = ["Team", "Actual", "LogReg", "NN"]
        if 'pred_xgb' in test_df.columns:
            cols_display.append('XGBoost')
        if 'pred_dt' in test_df.columns:
            cols_display.append('DT')
        if 'pred_rf' in test_df.columns:
            cols_display.append('RF')
        cols_display.append('Upset %')

        # Create a formatted header string
        header = f"{cols_display[0]:<30} {cols_display[1]:<8} {cols_display[2]:<8} {cols_display[3]:<8}"
        idx = 4
        while idx < len(cols_display) - 1:
            header += f" {cols_display[idx]:<8}"
            idx += 1
        header += f" {cols_display[-1]:<10}"

        width = 100 + 8 * (len(cols_display) - 5) if len(cols_display) > 5 else 100
        print("\nGAME PREDICTIONS FOR TEST DATA (with real outcomes):")
        print("-" * width)
        print(header)
        print("-" * width)
        for i, row in test_df.reset_index().iterrows():
            team = row['Team']
            actual = 'WIN' if row['actual'] == 1 else 'LOSS'
            lg = 'WIN' if row['pred_log'] == 1 else 'LOSS'
            nn = 'WIN' if row['pred_nn'] == 1 else 'LOSS'
            upset_pct = f"{row['Upset_Factor']*100:.1f}%" if ('Upset_Factor' in row and pd.notna(row['Upset_Factor'])) else "N/A"
            out = f"{team:<30} {actual:<8} {lg:<8} {nn:<8}"
            if 'pred_xgb' in row.index:
                xgbp = 'WIN' if row['pred_xgb'] == 1 else 'LOSS'
                out += f" {xgbp:<8}"
            if 'pred_dt' in row.index:
                dtp = 'WIN' if row['pred_dt'] == 1 else 'LOSS'
                out += f" {dtp:<8}"
            if 'pred_rf' in row.index:
                rfp = 'WIN' if row['pred_rf'] == 1 else 'LOSS'
                out += f" {rfp:<8}"
            out += f" {upset_pct:<10}"
            print(out)
        print("-" * (100 if XGBClassifier is None else 140))
        # Display accuracy scores for each model on the test data.
        # Accuracy = (correct_predictions / total_predictions).
        # For a balanced binary classification, 0.50 is random guessing; higher is better.
        print(f"Logistic Regression accuracy on test games: {log_acc:.4f}")
        print(f"Neural Network accuracy on test games:      {nn_acc:.4f}")
        if XGBClassifier is not None:
            print(f"XGBoost accuracy on test games:             {xgb_acc:.4f}")
        # Confusion matrices show the breakdown of predictions:
        # [[true_negatives, false_positives], [false_negatives, true_positives]]
        # Off-diagonal entries reveal where models go wrong.
        print("\nLogReg confusion matrix:\n", cm_log_test)
        print("\nNN confusion matrix:\n", cm_nn_test)
        if XGBClassifier is not None:
            print("\nXGBoost confusion matrix:\n", cm_xgb_test)

        # Print mismatches: identify rows where ANY model prediction differs from actual outcome.
        # Strategy: build a boolean mask by comparing each prediction column (pred_log, pred_nn,
        # pred_xgb, pred_dt, pred_rf) to the 'actual' column row-wise. If any prediction
        # differs from actual in that row, mark as a mismatch. This shows where the ensemble
        # of models failed and deserves investigation.
        cols_to_check = ['pred_log','pred_nn']
        if 'pred_xgb' in test_df.columns:
            cols_to_check.append('pred_xgb')
        if 'pred_dt' in test_df.columns:
            cols_to_check.append('pred_dt')
        if 'pred_rf' in test_df.columns:
            cols_to_check.append('pred_rf')
        # Row-wise comparison: test_df[cols_to_check].ne(..., axis=0) compares each row
        # element-wise; .any(axis=1) marks a row if any column differs from actual.
        mismatch_mask = test_df[cols_to_check].ne(test_df['actual'], axis=0).any(axis=1)
        mismatches = test_df.loc[mismatch_mask, ['Team','Wins','Losses','actual'] + cols_to_check]
        if not mismatches.empty:
            print('\nMismatches (teams where any model disagreed with actual outcome):')
            for _, r in mismatches.iterrows():
                # Build readable summary: show actual outcome and all model predictions
                parts = [f"{r['Team']}: actual={'WIN' if r['actual']==1 else 'LOSS'}"]
                parts.append(f"LogReg={'WIN' if r['pred_log']==1 else 'LOSS'}")
                parts.append(f"NN={'WIN' if r['pred_nn']==1 else 'LOSS'}")
                if XGBClassifier is not None:
                    parts.append(f"XGB={'WIN' if r['pred_xgb']==1 else 'LOSS'}")
                if 'pred_dt' in r.index:
                    parts.append(f"DT={'WIN' if r['pred_dt']==1 else 'LOSS'}")
                if 'pred_rf' in r.index:
                    parts.append(f"RF={'WIN' if r['pred_rf']==1 else 'LOSS'}")
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

    # Team-name resolution: handle partial matches and case insensitivity
    # Strategy:
    # 1. Try exact match (case-insensitive) for direct hits.
    # 2. Fall back to substring match if exactly one team contains the input.
    # 3. Return None if no match or multiple ambiguous matches.
    # This allows users to supply short names like 'Eagles', 'Chiefs' or
    # partial names like 'New' or 'Tampa' while still catching typos.
    available_teams = df['Team'].unique()
    def resolve_team(name):
        name = name.strip()
        # Exact match (case-insensitive)
        for t in available_teams:
            if name.lower() == t.lower():
                return t
        # Substring match (case-insensitive); only accept if unambiguous
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
    # Interactive team selection with validation loop.
    # Continues until the user provides two different, unambiguous team names.
    # The loop offers helpful error messages for common issues:
    # - No match found (typo or incorrect abbreviation).
    # - Multiple matches (ambiguous input; user must be more specific).
    # - Same team selected twice (nonsensical matchup).
    # - Keyboard interrupt (user can Ctrl+C to exit gracefully).
    while True:
        try:
            team1_input = input("\nEnter first team name (or part of it): ").strip()
            team2_input = input("Enter second team name (or part of it): ").strip()
            
            # Find matching teams (case-insensitive substring match)
            team1_matches = [t for t in available_teams if team1_input.lower() in t.lower()]
            team2_matches = [t for t in available_teams if team2_input.lower() in t.lower()]
            
            # Validate team1 match
            if not team1_matches:
                print(f"No team found matching '{team1_input}'. Please try again.")
                continue
            if len(team1_matches) > 1:
                print(f"Multiple matches for '{team1_input}': {', '.join(team1_matches)}")
                continue
            
            # Validate team2 match
            if not team2_matches:
                print(f"No team found matching '{team2_input}'. Please try again.")
                continue
            if len(team2_matches) > 1:
                print(f"Multiple matches for '{team2_input}': {', '.join(team2_matches)}")
                continue
            
            # Extract unique matches and validate they differ
            team1 = team1_matches[0]
            team2 = team2_matches[0]
            
            if team1 == team2:
                print("Please enter two different teams.")
                continue
            
            # Valid pair: exit the loop and proceed to predictions
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
# Make per-model probability comparisons so each model chooses exactly one winner
# Logistic Regression probabilities and winner
try:
    log_reg_p1 = float(log_reg.predict_proba(team1_scaled)[0][1])
    log_reg_p2 = float(log_reg.predict_proba(team2_scaled)[0][1])
except Exception:
    # fallback to predicted class if predict_proba unavailable
    log_reg_p1 = float(log_reg.predict(team1_scaled)[0])
    log_reg_p2 = float(log_reg.predict(team2_scaled)[0])
log_reg_winner = team1 if log_reg_p1 > log_reg_p2 else team2

# Neural network probabilities and winner
try:
    nn_p1 = float(model.predict(team1_scaled, verbose=0)[0][0])
    nn_p2 = float(model.predict(team2_scaled, verbose=0)[0][0])
except Exception:
    nn_p1 = None
    nn_p2 = None
nn_winner = team1 if (nn_p1 is not None and nn_p2 is not None and nn_p1 > nn_p2) else team2

# XGBoost (if available)
if XGBClassifier is not None and 'xgb_clf' in globals() and xgb_clf is not None:
    try:
        xgb_p1 = float(xgb_clf.predict_proba(team1_scaled)[0][1])
        xgb_p2 = float(xgb_clf.predict_proba(team2_scaled)[0][1])
    except Exception:
        xgb_p1 = None
        xgb_p2 = None
    xgb_winner = team1 if (xgb_p1 is not None and xgb_p2 is not None and xgb_p1 > xgb_p2) else team2
else:
    xgb_p1 = xgb_p2 = None
    xgb_winner = None

# Decision Tree / Random Forest probabilities and winners (if available)
if 'dt_clf' in globals() and dt_clf is not None:
    try:
        dt_p1 = float(dt_clf.predict_proba(team1_scaled)[0][1])
        dt_p2 = float(dt_clf.predict_proba(team2_scaled)[0][1])
    except Exception:
        dt_p1 = None
        dt_p2 = None
    dt_winner = team1 if (dt_p1 is not None and dt_p2 is not None and dt_p1 > dt_p2) else team2
else:
    dt_p1 = dt_p2 = None
    dt_winner = None

if 'rf_clf' in globals() and rf_clf is not None:
    try:
        rf_p1 = float(rf_clf.predict_proba(team1_scaled)[0][1])
        rf_p2 = float(rf_clf.predict_proba(team2_scaled)[0][1])
    except Exception:
        rf_p1 = None
        rf_p2 = None
    rf_winner = team1 if (rf_p1 is not None and rf_p2 is not None and rf_p1 > rf_p2) else team2
else:
    rf_p1 = rf_p2 = None
    rf_winner = None

# Display results
print("\n" + "="*60)
print(f"MATCHUP: {team1} vs {team2}")
print("="*60)
print(f"\n{'Model':<25} {'Team 1 ({})': <30} {'Team 2 ({})': <30}")
print(f"{'': <25} {team1: <30} {team2: <30}")
print("-" * 85)
print(f"{'Logistic Regression': <25} {'WIN' if log_reg_winner == team1 else 'LOSS': <30} {'WIN' if log_reg_winner == team2 else 'LOSS': <30}")
print(f"{'Neural Network': <25} {'WIN' if nn_winner == team1 else 'LOSS': <30} {'WIN' if nn_winner == team2 else 'LOSS': <30}")
if XGBClassifier is not None:
    print(f"{'XGBoost': <25} {'WIN' if xgb_winner == team1 else 'LOSS': <30} {'WIN' if xgb_winner == team2 else 'LOSS': <30}")
print("-" * 85)

# (Upset percentages will be displayed after consensus is computed)

print(f"\nLogistic Regression predicts: {log_reg_winner} wins")
print(f"Neural Network predicts: {nn_winner} wins")
if XGBClassifier is not None:
    print(f"XGBoost predicts: {xgb_winner} wins")
if 'dt_clf' in globals() and dt_clf is not None:
    print(f"Decision Tree predicts: {dt_winner} wins")
if 'rf_clf' in globals() and rf_clf is not None:
    print(f"Random Forest predicts: {rf_winner} wins")

# Compute a simple vote-based consensus across available models.
#
# Each model casts one vote for the team it predicts will have a
# winning record. This is a majority-vote scheme; ties are awarded to
# `team1` by the current logic (>=). For a more sophisticated ensemble
# you could weight votes by held-out accuracy or use probability
# averaging; this implementation keeps the behavior transparent.
# Use probability comparisons so each model casts exactly one vote
votes_team1 = 0
votes_team2 = 0
votes_team1 += int(log_reg_p1 > log_reg_p2) if (log_reg_p1 is not None and log_reg_p2 is not None) else 0
votes_team2 += int(log_reg_p2 > log_reg_p1) if (log_reg_p1 is not None and log_reg_p2 is not None) else 0
votes_team1 += int(nn_p1 > nn_p2) if (nn_p1 is not None and nn_p2 is not None) else 0
votes_team2 += int(nn_p2 > nn_p1) if (nn_p1 is not None and nn_p2 is not None) else 0
if XGBClassifier is not None and xgb_p1 is not None and xgb_p2 is not None:
    votes_team1 += int(xgb_p1 > xgb_p2)
    votes_team2 += int(xgb_p2 > xgb_p1)
if 'dt_clf' in globals() and dt_p1 is not None and dt_p2 is not None:
    votes_team1 += int(dt_p1 > dt_p2)
    votes_team2 += int(dt_p2 > dt_p1)
if 'rf_clf' in globals() and rf_p1 is not None and rf_p2 is not None:
    votes_team1 += int(rf_p1 > rf_p2)
    votes_team2 += int(rf_p2 > rf_p1)

final_winner = team1 if votes_team1 >= votes_team2 else team2

if log_reg_winner == nn_winner:
    print(f"\nAll models agree: {log_reg_winner} wins this matchup!")
else:
    if XGBClassifier is not None:
        consensus = team1 if votes_team1 > votes_team2 else team2
        print(f"\nModels disagree. 3-model vote: {consensus} wins ({votes_team1} vs {votes_team2})")
    else:
        print(f"\nModels disagree - it's a close call! (vote: {votes_team1} vs {votes_team2})")

# After determining the consensus winner, display the upset chance for the
# losing team only — the chance an upset could occur against the predicted winner.
loser = team2 if final_winner == team1 else team1

# Gather upset factor for the losing team (from the training dataframe `df`)
loser_upset_val = None
if 'Upset_Factor' in df.columns:
    s = df.loc[df['Team'] == loser, 'Upset_Factor']
    if not s.empty and pd.notna(s.values[0]):
        # Normalize Upset_Factor: if it's >1 assume it's a percentage (e.g. 20 -> 20%)
        raw = float(s.values[0])
        if raw > 1.0:
            loser_upset_val = raw / 100.0
        else:
            loser_upset_val = raw
        # Clamp to [0,1]
        loser_upset_val = max(0.0, min(1.0, loser_upset_val))

# Build an ensemble probability for each team by averaging available model
# predicted probabilities for the "winning" class (class 1). Fall back
# gracefully if some models are not present.
probs_team1 = []
probs_team2 = []
if log_reg_p1 is not None and log_reg_p2 is not None:
    probs_team1.append(log_reg_p1)
    probs_team2.append(log_reg_p2)
if nn_p1 is not None and nn_p2 is not None:
    probs_team1.append(nn_p1)
    probs_team2.append(nn_p2)
if XGBClassifier is not None and xgb_p1 is not None and xgb_p2 is not None:
    probs_team1.append(xgb_p1)
    probs_team2.append(xgb_p2)
if 'dt_clf' in globals() and dt_p1 is not None and dt_p2 is not None:
    probs_team1.append(dt_p1)
    probs_team2.append(dt_p2)
if 'rf_clf' in globals() and rf_p1 is not None and rf_p2 is not None:
    probs_team1.append(rf_p1)
    probs_team2.append(rf_p2)

ensemble_p1 = float(sum(probs_team1) / len(probs_team1)) if len(probs_team1) > 0 else None
ensemble_p2 = float(sum(probs_team2) / len(probs_team2)) if len(probs_team2) > 0 else None

# Determine underdog using an odds-based head-to-head conversion.
# Convert per-team ensemble probs to odds and compute a head-to-head
# base probability so the two probabilities form a proper pairwise
# distribution. This avoids treating both high ensemble probs as
# independent head-to-head chances.
underdog = None
underdog_prob = None
favorite_prob = None
hh_p1 = hh_p2 = None
if ensemble_p1 is not None and ensemble_p2 is not None:
    try:
        def safe_odds(p):
            p = max(1e-6, min(1.0 - 1e-6, float(p)))
            return p / (1.0 - p)
        o1 = safe_odds(ensemble_p1)
        o2 = safe_odds(ensemble_p2)
        hh_p1 = o1 / (o1 + o2)
        hh_p2 = 1.0 - hh_p1
        # underdog is the team with the smaller head-to-head probability
        if hh_p1 < hh_p2:
            underdog = team1
            underdog_prob = hh_p1
            favorite_prob = hh_p2
        else:
            underdog = team2
            underdog_prob = hh_p2
            favorite_prob = hh_p1
    except Exception:
        # fallback to the simpler approach if something goes wrong
        if ensemble_p1 < ensemble_p2:
            underdog = team1
            underdog_prob = ensemble_p1
            favorite_prob = ensemble_p2
        else:
            underdog = team2
            underdog_prob = ensemble_p2
            favorite_prob = ensemble_p1

# Get the losing team's win ratio from training records (Wins / (Wins+Losses))
win_ratio = None
wr = df.loc[df['Team'] == loser, ['Wins', 'Losses']]
if not wr.empty:
    try:
        w = float(wr.iloc[0]['Wins'])
        l = float(wr.iloc[0]['Losses'])
        if (w + l) > 0:
            win_ratio = w / (w + l)
    except Exception:
        win_ratio = None

# Compute an adjusted upset probability combining multiple signals:
#
# The adjusted upset probability blends:
# 1. Underdog ensemble probability (how likely models think underdog wins)
# 2. Upset_Factor (domain knowledge: historical volatility for this team)
# 3. Inverse win ratio (teams with poor records are less likely to upset)
# 4. Strength gap (how close the matchup is; closer = higher upset chance)
#
# Weighting rationale:
# - Upset_Factor is weighted highest (alpha=0.6) because it is domain-derived
#   and captures long-term team tendencies and volatility.
# - Inverse win ratio (beta=0.3) adds an empirical component: teams with poor
#   records have been losing consistently and are unlikely to suddenly win.
# - Strength gap (gamma=0.1) is low weight because we're already accounting
#   for strength via ensemble probability.
#
# Formula: adjusted_upset = underdog_prob * modifier, where
#   modifier = (0.6 * upset_factor + 0.3 * (1-win_ratio) + 0.1 * (1-gap))
# This scales the underdog's base probability by a domain-aware factor.
# Final result is clamped to [0, 1] to stay within valid probability range.
adjusted_upset = None
if loser_upset_val is not None:
    if underdog_prob is not None and favorite_prob is not None:
        # Normalize strength gap to [0, 1]: gap = 0 means equally strong teams.
        strength_gap = max(0.0, min(1.0, abs(favorite_prob - underdog_prob)))
        alpha = 0.6  # weight for Upset_Factor (domain volatility)
        beta = 0.3   # weight for inverse win ratio (empirical performance)
        gamma = 0.1  # weight for closeness (1 - strength_gap)
        inv_win = (1.0 - win_ratio) if win_ratio is not None else 0.5
        # Combine signals: higher upset_factor, lower win_ratio, and
        # tighter matchup all increase upset likelihood.
        modifier = (alpha * loser_upset_val) + (beta * inv_win) + (gamma * (1.0 - strength_gap))
        # Optional scale to increase upset sensitivity (tunable)
        upset_scale = 1.5
        modifier *= upset_scale
        # Scale base underdog probability by the combined modifier
        adjusted_upset = underdog_prob * modifier
    else:
        # If ensemble probs aren't available, fall back to a conservative
        # estimate using only Upset_Factor and win ratio (no model confidence).
        inv_win = (1.0 - win_ratio) if win_ratio is not None else 0.5
        adjusted_upset = 0.35 * loser_upset_val + 0.15 * inv_win

if adjusted_upset is not None:
    # Clamp to [0,1]
    adjusted_upset = max(0.0, min(1.0, float(adjusted_upset)))
    # Cap upset probability at a conservative upper bound so "upset"
    # remains the less-likely outcome when the ensemble favors the winner.
    max_upset_cap = 0.60
    capped_upset = min(adjusted_upset, max_upset_cap)

    # Print header label using the requested wording and show component breakdown
    label = 'Chances of the Losing Team to Upset the Winning Team:'
    upset_display = capped_upset * 100.0
    if final_winner == team1:
        print(f"{label: <25} {'':<30} {upset_display:.1f}%")
    else:
        print(f"{label: <25} {upset_display:.1f}%")

    # Provide supporting numbers for transparency
        if ensemble_p1 is not None and ensemble_p2 is not None:
            print(f"{'': <25} Ensemble probs -> {team1}: {ensemble_p1*100:.1f}%, {team2}: {ensemble_p2*100:.1f}%")
            if hh_p1 is not None and hh_p2 is not None:
                print(f"{'': <25} Head-to-head base -> {team1}: {hh_p1*100:.1f}%, {team2}: {hh_p2*100:.1f}%")
    if loser_upset_val is not None:
        print(f"{'': <25} Normalized Upset_Factor (dataset): {loser_upset_val:.2f}  (values >1 are treated as percentages and divided by 100)")
    # Show upset scale if available
    try:
        print(f"{'': <25} Upset scale applied: x{upset_scale:.2f}")
    except Exception:
        pass
    if win_ratio is not None:
        print(f"{'': <25} Losing team win ratio: {win_ratio*100:.1f}%")
    # Show the intermediate signals used in the modifier when available
    try:
        print(f"{'': <25} Modifier components -> alpha*Upset: {alpha*loser_upset_val:.3f}, beta*(1-win): {beta*inv_win:.3f}, gamma*(1-gap): {gamma*(1.0-strength_gap):.3f}")
    except Exception:
        pass
    print("-" * 85)

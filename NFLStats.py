# pip install pandas numpy scikit-learn tensorflow
# pip install xgboost
import xgboost as xgb
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from keras import layers, models, callbacks
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


Offense = pd.read_csv("Offense.csv")
Defense = pd.read_csv("Defense.csv")
SOS = pd.read_csv("SOS.csv")
Turnovers = pd.read_csv("Turnovers.csv")
Record = pd.read_csv("WinsLosses.csv")
HomeFieldAdv = pd.read_csv("HomeFieldAdv.csv")
WeatherEffects = pd.read_csv("WeatherEffects.csv")
UpsetFactor = pd.read_csv("UpsetFactor.csv")

# Standardize team names by stripping whitespace
Offense['Team'] = Offense['Team'].str.strip()
Defense['Team'] = Defense['Team'].str.strip()
SOS['Team'] = SOS['Team'].str.strip()
Turnovers['Team'] = Turnovers['Team'].str.strip()
Record['Team'] = Record['Team'].str.strip()
HomeFieldAdv['Team'] = HomeFieldAdv['Team'].str.strip()
WeatherEffects['Team'] = WeatherEffects['Team'].str.strip()
UpsetFactor['Team'] = UpsetFactor['Team'].str.strip()

# Map short names in Record to full names in other files
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

# Map team names and strip whitespace from column names
Record['Team'] = Record['Team'].map(team_mapping)
Record.columns = Record.columns.str.strip()

# Strip whitespace from all column names in all dataframes
Offense.columns = Offense.columns.str.strip()
Defense.columns = Defense.columns.str.strip()
SOS.columns = SOS.columns.str.strip()
Turnovers.columns = Turnovers.columns.str.strip()

df = Offense.merge(Defense, on="Team", suffixes=('_off', '_def'))
df = df.merge(SOS, on="Team")
df = df.merge(Turnovers, on="Team")
df = df.merge(Record, on="Team")

# Merge home field advantage (take Rank as feature)
df = df.merge(HomeFieldAdv[['Team', 'Rank']], on='Team', how='left')
df.rename(columns={'Rank': 'HomeFieldAdv_Rank'}, inplace=True)

# Merge weather effects
df = df.merge(WeatherEffects[['Team', 'Weather_Difficulty_Rating']], on='Team', how='left')

# Merge upset factor
df = df.merge(UpsetFactor[['Team', 'Upset_Factor']], on='Team', how='left')

# Create target: 1 if Wins > Losses (winning record), 0 otherwise
df['target'] = (df['Wins'] > df['Losses']).astype(int)

# Features: drop Team, Wins, Losses, OCR, target, and non-numeric columns
x = df.drop(columns=["Team", "Wins", "Losses", "OCR", "target", "Stadium"], errors='ignore')
y = df['target'] # Winner target

# Debug: check for non-numeric columns
print("Data types:")
print(x.dtypes)
print("\nDataframe info:")
print(x.info())

# Check for NaN values
if df.isna().any().any():
    print("\nWARNING: NaN values found in dataframe!")
    print(df.isna().sum())
    print("\nTeams with NaN values:")
    print(df[df.isna().any(axis=1)][['Team', 'Wins', 'Losses']])
else:
    print("\nNo NaN values found. Data looks good!")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42) # 75% train, 25% test
scaler = StandardScaler() # Standardize features
x_train = scaler.fit_transform(x_train) # Fit on training data
x_test = scaler.transform(x_test) # Transform test data
le = LabelEncoder() # Encode target labels
y_train = le.fit_transform(y_train) # Fit and transform training labels
y_test = le.transform(y_test) # Transform test labels

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000) # Increase max_iter
log_reg.fit(x_train, y_train) # Train model
y_pred_log_reg = log_reg.predict(x_test) # Predict on test data
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg) # Confusion matrix
print("Logistic Regression Confusion Matrix:") 
print(cm_log_reg) 
# XGBoost (if available)
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
# Neural Network Model
model = models.Sequential() 
model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],))) # Input layer
model.add(layers.Dense(32, activation='relu')) # Hidden layer
model.add(layers.Dense(1, activation='sigmoid')) # Output layer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Compile model
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # Early stopping
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.15, callbacks=[early_stopping], verbose=0) # Train model
y_pred_nn = (model.predict(x_test, verbose=0) > 0.5).astype("int32") # Predict on test data
cm_nn = confusion_matrix(y_test, y_pred_nn) # Confusion matrix
print("Neural Network Confusion Matrix:")
print(cm_nn)

# ===== GAME PREDICTION =====
print("\n" + "="*60)
print("PREDICTING GAMES WITH UNSEEN DATA")
print("="*60)

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
    
    # Strip whitespace from test data column names
    OffenseTest.columns = OffenseTest.columns.str.strip()
    DefenseTest.columns = DefenseTest.columns.str.strip()
    SOSTest.columns = SOSTest.columns.str.strip()
    TurnoversTest.columns = TurnoversTest.columns.str.strip()
    HomeFieldAdvTest.columns = HomeFieldAdvTest.columns.str.strip()
    WeatherEffectsTest.columns = WeatherEffectsTest.columns.str.strip()
    
    # Map team names in test data
    OffenseTest['Team'] = OffenseTest['Team'].map(team_mapping)
    DefenseTest['Team'] = DefenseTest['Team'].map(team_mapping)
    SOSTest['Team'] = SOSTest['Team'].map(team_mapping)
    TurnoversTest['Team'] = TurnoversTest['Team'].map(team_mapping)
    HomeFieldAdvTest['Team'] = HomeFieldAdvTest['Team'].map(team_mapping)
    WeatherEffectsTest['Team'] = WeatherEffectsTest['Team'].map(team_mapping)
    UpsetFactorTest['Team'] = UpsetFactorTest['Team'].map(team_mapping)
    
    # Merge test data the same way as training data
    test_df = OffenseTest.merge(DefenseTest, on="Team", suffixes=('_off', '_def'))
    test_df = test_df.merge(SOSTest, on="Team")
    test_df = test_df.merge(TurnoversTest, on="Team")
    test_df = test_df.merge(HomeFieldAdvTest[['Team', 'Rank']], on='Team', how='left')
    test_df.rename(columns={'Rank': 'HomeFieldAdv_Rank'}, inplace=True)
    test_df = test_df.merge(WeatherEffectsTest[['Team', 'Weather_Difficulty_Rating']], on='Team', how='left')
    test_df = test_df.merge(UpsetFactorTest[['Team', 'Upset_Factor']], on='Team', how='left')
    
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

    # attach predictions to dataframe for easy validation
    test_df['pred_log'] = log_reg_predictions
    test_df['pred_nn'] = nn_predictions

    # If a Wins/Losses test file exists, merge it to compute real outcomes and validate
    try:
        WinsTest = pd.read_csv("WinsLossesTest.csv")
        WinsTest['Team'] = WinsTest['Team'].str.strip()
        WinsTest.columns = WinsTest.columns.str.strip()
        # Map short names to full names (same mapping used earlier)
        WinsTest['Team'] = WinsTest['Team'].map(team_mapping)

        # Merge into test_df to get actual results
        test_df = test_df.merge(WinsTest, on='Team', how='left')
        # Compute actual label
        test_df['actual'] = (test_df['Wins'] > test_df['Losses']).astype(int)

        # Compare predictions to actuals
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

        # Print mismatches
        cols_to_check = ['pred_log','pred_nn'] + (['pred_xgb'] if XGBClassifier is not None else [])
        mismatches = test_df.loc[~(test_df['actual'] == test_df[cols_to_check]).all(axis=1), ['Team','Wins','Losses','actual'] + cols_to_check]
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
print("\n" + "="*60)
print("PREDICT WINNER BETWEEN TWO TEAMS")
print("="*60)

# Get available teams from the training data
available_teams = df['Team'].unique()
print(f"\nAvailable teams ({len(available_teams)}):")
for i, team in enumerate(sorted(available_teams), 1):
    print(f"  {i}. {team}")

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

# Get upset factors for display
team1_upset = df[df['Team'] == team1]['Upset_Factor'].values[0] if 'Upset_Factor' in df.columns else None
team2_upset = df[df['Team'] == team2]['Upset_Factor'].values[0] if 'Upset_Factor' in df.columns else None

if team1_upset is not None and team2_upset is not None:
    print(f"{'Chances of Upset:': <25} {team1_upset*100:.1f}% {'':<24} {team2_upset*100:.1f}%")
    print("-" * 85)

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

if log_reg_winner == nn_winner:
    print(f"\nBoth models agree: {log_reg_winner} wins this matchup!")
else:
    # determine 3-model consensus if XGBoost available
    if XGBClassifier is not None:
        votes_team1 = int(log_reg_team1) + int(nn_team1) + int(xgb_team1)
        votes_team2 = int(log_reg_team2) + int(nn_team2) + int(xgb_team2)
        consensus = team1 if votes_team1 > votes_team2 else team2
        print(f"\nModels disagree. 3-model vote: {consensus} wins ({votes_team1} vs {votes_team2})")
    else:
        print(f"\nModels disagree - it's a close call!")

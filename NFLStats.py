# pip install pandas numpy scikit-learn tensorflow

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from keras import layers, models, callbacks


Offense = pd.read_csv("Offense.csv")
Defense = pd.read_csv("Defense.csv")
SOS = pd.read_csv("SOS.csv")
Turnovers = pd.read_csv("Turnovers.csv")
Record = pd.read_csv("WinsLosses.csv")

# Standardize team names by stripping whitespace
Offense['Team'] = Offense['Team'].str.strip()
Defense['Team'] = Defense['Team'].str.strip()
SOS['Team'] = SOS['Team'].str.strip()
Turnovers['Team'] = Turnovers['Team'].str.strip()
Record['Team'] = Record['Team'].str.strip()

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

# Create target: 1 if Wins > Losses (winning record), 0 otherwise
df['target'] = (df['Wins'] > df['Losses']).astype(int)

# Features: drop Team, Wins, Losses, OCR, and target (target is the label, not a feature)
x = df.drop(columns=["Team", "Wins", "Losses", "OCR", "target"])
y = df['target']

# Debug: check for non-numeric columns
print("Data types:")
print(x.dtypes)
print("\nDataframe info:")
print(x.info())

# Check for NaN values
if df.isna().any().any():
    print("\n⚠️  WARNING: NaN values found in dataframe!")
    print(df.isna().sum())
    print("\nTeams with NaN values:")
    print(df[df.isna().any(axis=1)][['Team', 'Wins', 'Losses']])
else:
    print("\n✓ No NaN values found. Data looks good!")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train, y_train)
y_pred_log_reg = log_reg.predict(x_test)
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
print("Logistic Regression Confusion Matrix:")
print(cm_log_reg)
# Neural Network Model
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.15, callbacks=[early_stopping])
y_pred_nn = (model.predict(x_test) > 0.5).astype("int32")
cm_nn = confusion_matrix(y_test, y_pred_nn)
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
    
    # Standardize test data team names
    OffenseTest['Team'] = OffenseTest['Team'].str.strip()
    DefenseTest['Team'] = DefenseTest['Team'].str.strip()
    SOSTest['Team'] = SOSTest['Team'].str.strip()
    TurnoversTest['Team'] = TurnoversTest['Team'].str.strip()
    
    # Strip whitespace from test data column names
    OffenseTest.columns = OffenseTest.columns.str.strip()
    DefenseTest.columns = DefenseTest.columns.str.strip()
    SOSTest.columns = SOSTest.columns.str.strip()
    TurnoversTest.columns = TurnoversTest.columns.str.strip()
    
    # Map team names in test data
    OffenseTest['Team'] = OffenseTest['Team'].map(team_mapping)
    DefenseTest['Team'] = DefenseTest['Team'].map(team_mapping)
    SOSTest['Team'] = SOSTest['Team'].map(team_mapping)
    TurnoversTest['Team'] = TurnoversTest['Team'].map(team_mapping)
    
    # Merge test data the same way as training data
    test_df = OffenseTest.merge(DefenseTest, on="Team", suffixes=('_off', '_def'))
    test_df = test_df.merge(SOSTest, on="Team")
    test_df = test_df.merge(TurnoversTest, on="Team")
    
    # Prepare features for prediction (same columns as training, in same order)
    x_predict = test_df.drop(columns=["Team", "OCR"], errors='ignore')
    
    # Ensure test data has the same columns in the same order as training
    x_predict = x_predict[x.columns]
    
    # Scale using the same scaler from training
    x_predict_scaled = scaler.transform(x_predict)
    
    # Make predictions with both models
    log_reg_predictions = log_reg.predict(x_predict_scaled)
    nn_predictions = (model.predict(x_predict_scaled) > 0.5).astype("int32").flatten()
    
    # Display results
    print("\nGAME PREDICTIONS FOR TEST DATA:")
    print("-" * 80)
    print(f"{'Team':<25} {'Logistic Reg':<20} {'Neural Network':<20}")
    print("-" * 80)
    
    for i, team in enumerate(test_df['Team']):
        log_pred = "WIN (1)" if log_reg_predictions[i] == 1 else "LOSS (0)"
        nn_pred = "WIN (1)" if nn_predictions[i] == 1 else "LOSS (0)"
        print(f"{team:<25} {log_pred:<20} {nn_pred:<20}")
    
    print("-" * 80)
    
    # Summary statistics
    print("\nMODEL ACCURACY ON TEST DATA:")
    print(f"Logistic Regression Accuracy: {log_reg.score(x_test, y_test):.4f}")
    nn_accuracy = np.mean(y_pred_nn == y_test)
    print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
    
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

# Display results
print("\n" + "="*60)
print(f"MATCHUP: {team1} vs {team2}")
print("="*60)
print(f"\n{'Model':<25} {'Team 1 ({})': <30} {'Team 2 ({})': <30}")
print(f"{'': <25} {team1: <30} {team2: <30}")
print("-" * 85)
print(f"{'Logistic Regression': <25} {'WIN' if log_reg_team1 == 1 else 'LOSS': <30} {'WIN' if log_reg_team2 == 1 else 'LOSS': <30}")
print(f"{'Neural Network': <25} {'WIN' if nn_team1 == 1 else 'LOSS': <30} {'WIN' if nn_team2 == 1 else 'LOSS': <30}")
print("-" * 85)

# Determine consensus winner
log_reg_winner = team1 if log_reg_team1 > log_reg_team2 else team2
nn_winner = team1 if nn_team1 > nn_team2 else team2

print(f"\nLogistic Regression predicts: {log_reg_winner} wins")
print(f"Neural Network predicts: {nn_winner} wins")

if log_reg_winner == nn_winner:
    print(f"\nBoth models agree: {log_reg_winner} wins this matchup!")
else:
    print(f"\nModels disagree - it's a close call!")

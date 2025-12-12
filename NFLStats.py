# pip install pandas numpy scikit-learn tensorflow requests

import tensorflow as tf
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from keras import layers, models, callbacks


# load team stats from CSV files
Offense = pd.read_csv("Offense.csv")  # offense stats
Defense = pd.read_csv("Defense.csv")  # defense stats
SOS = pd.read_csv("SOS.csv")  # schedule strength
Turnovers = pd.read_csv("Turnovers.csv")  # turnover data
Record = pd.read_csv("WinsLosses.csv")  # team records

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

# combine all data into one dataframe
df = Offense.merge(Defense, on="Team", suffixes=('_off', '_def'))  # merge offense + defense, add suffix to duplicate columns
df = df.merge(SOS, on="Team")  # add schedule strength
df = df.merge(Turnovers, on="Team")  # add turnover stats
df = df.merge(Record, on="Team")  # add wins/losses

# Create target: 1 if Wins > Losses (winning record), 0 otherwise
df['target'] = (df['Wins'] > df['Losses']).astype(int)

# Features: drop Team, Wins, Losses, OCR, and target (target is the label, not a feature)
x = df.drop(columns=["Team", "Wins", "Losses", "OCR", "target"])
y = df['target'] # Winner target

# Debug: check for non-numeric columns
print("Data types:")
print(x.dtypes)
print("\nDataframe info:")
print(x.info())

# check for missing data
if df.isna().any().any():
    print("WARNING: NaN values found in dataframe!")  # missing data detected
    print(df.isna().sum())  # show how many missing per column
    print("Teams with NaN values:")  # which teams have problems
    print(df[df.isna().any(axis=1)][['Team', 'Wins', 'Losses']])
else:
    print("No NaN values found")  # all data looks good

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)  # split: 75% train, 25% test
scaler = StandardScaler()  # normalize features (mean 0, std 1)
x_train = scaler.fit_transform(x_train)  # fit + transform training data
x_test = scaler.transform(x_test)  # transform test using training params
le = LabelEncoder()  # convert labels to numbers
y_train = le.fit_transform(y_train)  # fit + transform training labels
y_test = le.transform(y_test)  # transform test labels

# logistic regression - simple fast model
log_reg = LogisticRegression(max_iter=1000)  # iterations to converge
log_reg.fit(x_train, y_train)  # train on normalized data
y_pred_log_reg = log_reg.predict(x_test)  # predictions on test set
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)  # accuracy breakdown
print("Logistic Regression Confusion Matrix:")
print(cm_log_reg)  # show results 
# neural network - complex model, better accuracy potential
model = models.Sequential()  # layer by layer
model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))  # 64 neurons input layer
model.add(layers.Dense(32, activation='relu'))  # 32 neurons hidden layer
model.add(layers.Dense(1, activation='sigmoid'))  # 1 neuron output, gives probability 0-1
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # setup training
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # stop if overfitting
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.15, callbacks=[early_stopping])  # train
y_pred_nn = (model.predict(x_test) > 0.5).astype("int32")  # convert probabilities to 0 or 1
cm_nn = confusion_matrix(y_test, y_pred_nn)  # accuracy breakdown
print("Neural Network Confusion Matrix:")
print(cm_nn)  # show results

# ===== GAME PREDICTION =====
print("\n" + "="*60)
print("PREDICTING GAMES WITH UNSEEN DATA")
print("="*60)

# load test data to make predictions on new games
try:
    OffenseTest = pd.read_csv("OffenseTest.csv")  # test offense stats
    DefenseTest = pd.read_csv("DefenseTest.csv")  # test defense stats
    SOSTest = pd.read_csv("SOSTest.csv")  # test schedule
    TurnoversTest = pd.read_csv("TurnoversTest.csv")  # test turnovers
    
    # Standardize test data team names
    OffenseTest['Team'] = OffenseTest['Team'].str.strip()  # remove spaces
    DefenseTest['Team'] = DefenseTest['Team'].str.strip()  # remove spaces
    SOSTest['Team'] = SOSTest['Team'].str.strip()  # remove spaces
    TurnoversTest['Team'] = TurnoversTest['Team'].str.strip()  # remove spaces
    
    # Strip whitespace from test data column names
    OffenseTest.columns = OffenseTest.columns.str.strip()
    DefenseTest.columns = DefenseTest.columns.str.strip()
    SOSTest.columns = SOSTest.columns.str.strip()
    TurnoversTest.columns = TurnoversTest.columns.str.strip()
    
    # Map team names in test data
    OffenseTest['Team'] = OffenseTest['Team'].map(team_mapping)  # short to full names
    DefenseTest['Team'] = DefenseTest['Team'].map(team_mapping)
    SOSTest['Team'] = SOSTest['Team'].map(team_mapping)
    TurnoversTest['Team'] = TurnoversTest['Team'].map(team_mapping)
    
    # Merge test data the same way as training data
    test_df = OffenseTest.merge(DefenseTest, on="Team", suffixes=('_off', '_def'))  # merge offense + defense
    test_df = test_df.merge(SOSTest, on="Team")  # add schedule
    test_df = test_df.merge(TurnoversTest, on="Team")  # add turnovers
    
    # Prepare features for prediction (same columns as training, in same order)
    x_predict = test_df.drop(columns=["Team", "OCR"], errors='ignore')  # drop non-feature columns
    
    # Ensure test data has the same columns in the same order as training
    x_predict = x_predict[x.columns]  # reorder + select correct columns
    
    # Scale using the same scaler from training
    x_predict_scaled = scaler.transform(x_predict)  # normalize
    
    # Make predictions with both models
    log_reg_predictions = log_reg.predict(x_predict_scaled)  # logistic predictions
    nn_predictions = (model.predict(x_predict_scaled) > 0.5).astype("int32").flatten()  # neural network predictions
    
    # Display results
    print("GAME PREDICTIONS FOR TEST DATA:")
    print("-" * 80)
    print(f"{'Team':<25} {'Logistic Reg':<20} {'Neural Network':<20}")
    print("-" * 80)
    
    for i, team in enumerate(test_df['Team']):
        log_pred = "WIN (1)" if log_reg_predictions[i] == 1 else "LOSS (0)"
        nn_pred = "WIN (1)" if nn_predictions[i] == 1 else "LOSS (0)"
        #print(f"{team:<25} {log_pred:<20} {nn_pred:<20}")
    
    print("-" * 80)
    
    # show accuracy
    print("MODEL ACCURACY ON TEST DATA:")
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

# ===== LIVE STATS FETCHING FUNCTIONS =====
def fetch_live_team_stats(team_name):
    # fetch real time team data from ESPN API
    try:
        print(f"\nFetching live stats for {team_name}...")
        
        # ESPN API endpoint for teams
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
        response = requests.get(url, timeout=5)  # get data with 5 second timeout
        response.raise_for_status()  # check for errors
        
        teams_data = response.json()
        espn_teams = teams_data['sports'][0]['leagues'][0]['teams']
        
        # Find matching team
        matched_team = None
        for team in espn_teams:
            if team_name.lower() in team['displayName'].lower() or team['name'].lower() in team_name.lower():
                matched_team = team
                break
        
        if not matched_team:
            print(f"❌ Could not find {team_name} on ESPN")
            return None
        
        # Extract available stats
        stats = {
            'Team': team_name,
            'Record': matched_team.get('record', [{}])[0].get('summary', 'N/A'),
            'Wins': matched_team.get('record', [{}])[0].get('wins', 'N/A'),
            'Losses': matched_team.get('record', [{}])[0].get('losses', 'N/A')
        }
        
        print(f"✓ Found: {matched_team['displayName']} ({stats['Record']})")
        return stats
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching stats: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def get_live_team_features(team_name, training_data_df):
    # get team stats from training data (baseline for model)
    try:
        # Try to get from training data first
        team_row = training_data_df[training_data_df['Team'] == team_name]
        
        if len(team_row) == 0:
            print(f"❌ Team '{team_name}' not found in training data")
            return None
        
        # Extract features (same columns as training)
        features = team_row.drop(columns=["Team", "Wins", "Losses", "OCR", "target"], errors='ignore').values[0]
        
        print(f"✓ Loaded stats for {team_name}")
        return features
        
    except Exception as e:
        print(f"❌ Error loading team features: {e}")
        return None

# ===== TWO-TEAM MATCHUP PREDICTION =====
print()
print("PREDICT WINNER BETWEEN TWO TEAMS")
print()
print("Fetching live stats from ESPN...\n")

# Get available teams from the training data
available_teams = df['Team'].unique()  # teams we have data for
print(f"Available teams ({len(available_teams)}):")
for i, team in enumerate(sorted(available_teams), 1):
    print(f"  {i}. {team}")

# Get user input for two teams
while True:
    try:
        team1_input = input("\nEnter first team name (or part of it): ").strip()
        team2_input = input("Enter second team name (or part of it): ").strip()
        
        # Find matching teams (case-insensitive, partial match)
        team1_matches = [t for t in available_teams if team1_input.lower() in t.lower()]  # search
        team2_matches = [t for t in available_teams if team2_input.lower() in t.lower()]
        
        if not team1_matches:
            print(f"No team found matching '{team1_input}'. Please try again.")
            continue
        if not team2_matches:
            print(f"No team found matching '{team2_input}'. Please try again.")
            continue
        
        if len(team1_matches) > 1:
            print(f"Multiple matches for '{team1_input}': {', '.join(team1_matches)}")  # too many matches
            continue
        if len(team2_matches) > 1:
            print(f"Multiple matches for '{team2_input}': {', '.join(team2_matches)}")
            continue
        
        team1 = team1_matches[0]  # get first match
        team2 = team2_matches[0]
        
        if team1 == team2:
            print("Please enter two different teams.")  # need different teams
            continue
        
        break
    
    except KeyboardInterrupt:
        print("Exiting")
        exit()

# Fetch live stats for both teams
print("\n" + "-"*60)
print("FETCHING LIVE STATS...")
print("-"*60)

live_team1 = fetch_live_team_stats(team1)  # team1 from ESPN
live_team2 = fetch_live_team_stats(team2)  # team2 from ESPN

# Get team features from training data
print("\nLoading feature data...")
team1_stats = get_live_team_features(team1, df)  # team1 stats
team2_stats = get_live_team_features(team2, df)  # team2 stats

if team1_stats is None or team2_stats is None:
    print("\n❌ Error loading team features. Using training data as fallback.")
    team1_stats = df[df['Team'] == team1].drop(columns=["Team", "Wins", "Losses", "OCR", "target"]).values[0]
    team2_stats = df[df['Team'] == team2].drop(columns=["Team", "Wins", "Losses", "OCR", "target"]).values[0]

# Ensure stats are in the same order as training features
team1_stats = pd.DataFrame([team1_stats], columns=x.columns)  # correct column order
team2_stats = pd.DataFrame([team2_stats], columns=x.columns)

# Scale the stats
team1_scaled = scaler.transform(team1_stats)  # normalize
team2_scaled = scaler.transform(team2_stats)

# Make predictions with probability scores
log_reg_team1_prob = log_reg.predict_proba(team1_scaled)[0][1]  # team1 win prob
log_reg_team2_prob = log_reg.predict_proba(team2_scaled)[0][1]  # team2 win prob
nn_team1_prob = model.predict(team1_scaled, verbose=0)[0][0]  # neural net team1
nn_team2_prob = model.predict(team2_scaled, verbose=0)[0][0]  # neural net team2

# Determine winners based on highest probability (only one winner per model)
log_reg_winner = team1 if log_reg_team1_prob > log_reg_team2_prob else team2  # logistic winner
nn_winner = team1 if nn_team1_prob > nn_team2_prob else team2  # neural net winner

# Display results
print("\n" + "="*60)
print(f"MATCHUP: {team1} vs {team2}")
print("="*60)

# Show live stats if available
if live_team1 and live_team2:
    print(f"\n📊 LIVE STATS (from ESPN):")
    print(f"  {team1}: {live_team1.get('Record', 'N/A')}")
    print(f"  {team2}: {live_team2.get('Record', 'N/A')}")

print(f"\n{'Model':<25} {'Team 1':<30} {'Team 2':<30}")
print(f"{'': <25} {team1:<30} {team2:<30}")
print("-" * 85)
print(f"{'Logistic Regression': <25} {log_reg_team1_prob:.4f} {'(WINNER)' if log_reg_winner == team1 else '':<23} {log_reg_team2_prob:.4f} {'(WINNER)' if log_reg_winner == team2 else '':<23}")
print(f"{'Neural Network': <25} {nn_team1_prob:.4f} {'(WINNER)' if nn_winner == team1 else '':<23} {nn_team2_prob:.4f} {'(WINNER)' if nn_winner == team2 else '':<23}")
print("-" * 85)

print(f"\n🏆 PREDICTIONS:")
print(f"  Logistic Regression predicts: {log_reg_winner} wins")
print(f"  Neural Network predicts: {nn_winner} wins")

if log_reg_winner == nn_winner:
    print(f"\n✓ Both models agree: {log_reg_winner} wins this matchup!")
else:
    print(f"\n⚠ Models disagree - {log_reg_winner} vs {nn_winner}")
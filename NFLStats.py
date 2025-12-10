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

Record['Team'] = Record['Team'].map(team_mapping)

df = Offense.merge(Defense, on="Team", suffixes=('_off', '_def'))
df = df.merge(SOS, on="Team")
df = df.merge(Turnovers, on="Team")
df = df.merge(Record, on="Team")

# Create target: 1 if Wins > Losses (winning record), 0 otherwise
df['target'] = (df[' Wins'] > df[' Losses']).astype(int)

# Features: drop Team, Wins, Losses, OCR, and target (target is the label, not a feature)
x = df.drop(columns=["Team", " Wins", " Losses", " OCR", "target"])
y = df['target']

# Debug: check for non-numeric columns
print("Data types:")
print(x.dtypes)
print("\nDataframe info:")
print(x.info())

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
    x_predict = test_df.drop(columns=["Team", " OCR"], errors='ignore')
    
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

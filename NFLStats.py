# pip install pandas numpy scikit-learn tensorflow

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras import layers, models, callbacks


Offense = pd.read_csv("Offense.csv")
Defense = pd.read_csv("Defense.csv")
SOS = pd.read_csv("SOS.csv")
Turnovers = pd.read_csv("Turnovers.csv")

df = Offense.merge(Defense, on="Team")
df = df.merge(SOS, on="Team")
df = df.merge(Turnovers, on="Team")

X = df.drop(columns=["Team", "Target"])
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # number of features
    layers.Dense(1, activation="sigmoid")     # logistic regression
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2
)
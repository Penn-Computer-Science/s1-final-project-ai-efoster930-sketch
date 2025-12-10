# pip install pandas numpy scikit-learn tensorflow

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from keras import layers, models, callbacks


Offense = pd.read_csv("Offense.csv")
Defense = pd.read_csv("Defense.csv")
SOS = pd.read_csv("SOS.csv")
Turnovers = pd.read_csv("Turnovers.csv")
Record = pd.read_csv("WinsLosses.csv")

df = Offense.merge(Defense, on="Team")
df = df.merge(SOS, on="Team")
df = df.merge(Turnovers, on="Team")
df = df.merge(Record, on="Team")
df = df.dropna()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = models.Sequential(
    [
        layers.Input(shape=(x_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ]
)
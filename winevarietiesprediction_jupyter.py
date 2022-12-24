import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from datetime import datetime
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from os import environ
from pathlib import Path
from sklearn.utils import Bunch
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split

# 1. Laden der Daten
# Laden Sie die Daten in den Arbeitsspeicher, um die Daten anschließend zu verarbeiten. 
wine_bunch = load_wine()
print("1. Wine dataset loaded ...")

X = wine_bunch['data']
y = wine_bunch['target']

#2. Teilen der Daten in Training und Test
# Bitte achten Sie auf eine zufällige Aufteilung der Daten und eine Ausgeglichenheit der Klassen, dabei sollen 70% der Daten zum Training verwendet werden.
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)
print("1.1. Data classes balanced ...")

X_smote_train, X_smote_test, y_smote_train, y_smote_test = train_test_split(X_smote, y_smote, shuffle=True, train_size=0.7, random_state=42)
print("1.2. Data shuffled then splitted in 70% training and 30% test ...")

#3. Trainieren eines geeigneten Algorithmus
# Wählen Sie einen geeigneten Algorithmus aus, um die Sorten des Weins vorherzusagen. Trainieren Sie den Algorithmus. 
model = DecisionTreeClassifier()
print(f"3.1. {model}'s algorithm chosen to predict wine classes ...")

model.fit(X_smote_train, y_smote_train)
print(f"3.2. {model}'s algorithm trained ...")

# 4. Testen des Algorithmus auf den Testdaten
# Berechnen Sie mindestens ein Maß für die Genauigkeit der Vorhersage.
expected = X_smote_test
predicted = model.predict(X_smote_test)
print(f"4.1. {model}'s algorithm tested ...")

accuracy_score = metrics.accuracy_score(expected, predicted)
print(f"4.2.1 {model}'s algorithm accuracy score = {accuracy_score}")


f1_score = metrics.f1_score(expected, predicted, average=None)
print(f"4.2.2. {model}'s algorithm f1 score = {f1_score}")

# 5. Illustration
# Stellen Sie graphisch dar, wie viele Weine der jeweiligen Klasse richtig vorhergesagt wurden.
correct_predictions = [0, 0, 0]
for i in range(len(predicted)):
    if predicted[i] == expected[i]:
        correct_predictions[predicted[i]] += 1
fig = plt.figure()
fig.suptitle(model+" correct predictions illstration", fontsize=12)
axes = fig.add_axes([0,0,1,1])
classes = ['class 0', 'class 1', 'class 2']
axes.bar(classes,correct_predictions)
plt.savefig(Path('result') / file_name)
plt.show()

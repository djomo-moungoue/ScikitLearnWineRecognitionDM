"""
Projet : Analyser le jeu de données Scikit-learn sur le vin (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) et utiliser les propriétés pour prédire la variété du vin.
"""

import numpy as np
import pandas as pd
from sklearn import datasets

print("\n-------------------------------------------------------------------------------------------")
print("| Objectif du projet : Prédire les variétés de vins du jeu de données de vin scikit-learn |")
print("-------------------------------------------------------------------------------------------\n")


# Charger/importer les données (1)
wine = datasets.load_wine()
print(wine)

# Clean the data

# Divisez les données en ensembles de formation et de test (2)
# Veillez à ce que les données soient divisées de manière aléatoire et que les classes soient équilibrées. 70% des données doivent être utilisées pour la formation.

# Créer un modèle

# Entraîner (le modèle) un algorithme approprié (3)
# Sélectionnez un algorithme approprié pour prédire les variétés de vin. Entraînez l'algorithme.

# Tester l'algorithme sur les données de test. (4)
# Calculez au moins une mesure de l'exactitude de la prédiction.

# Faites des prédictions

# Illustrez votre résultat (5)
# Illustrez graphiquement le nombre de vins de chaque classe qui ont été correctement prédits. 

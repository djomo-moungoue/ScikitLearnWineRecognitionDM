"""
Objectif : Analyser le jeu de données Scikit-learn sur le vin (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) et utiliser les propriétés pour prédire la variété du vin.
"""

import numpy as np
import pandas as pd
from sklearn import datasets

print("\n-------------------------------------------------------------------------------------------")
print("| Objectif du projet : Prédire les variétés de vins du jeu de données de vin scikit-learn |")
print("-------------------------------------------------------------------------------------------\n")


wine = datasets.load_wine()
print(wine)
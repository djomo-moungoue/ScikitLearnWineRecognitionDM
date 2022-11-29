"""
Projet : Analyser le jeu de données Scikit-learn sur le vin (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) et utiliser les propriétés pour prédire la variété du vin.
"""

from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
import math

print("\n-------------------------------------------------------------------------------------------")
print("| Objectif du projet : Prédire les variétés de vins du jeu de données de vin scikit-learn |")
print("-------------------------------------------------------------------------------------------\n")


# Charger/importer les données (1)

# Utilisez les jeux de données pour charger le jeu de données intégré sur le vin.
wine = datasets.load_wine(return_X_y=False, as_frame=False)
# print(wine)

# Divisez les données en ensembles de formation et de test (2)
# Veillez à ce que les données soient divisées de manière aléatoire et que les classes soient équilibrées. 70% des données doivent être utilisées pour la formation.

# Créez les objets X et y pour stocker respectivement les données et la valeur cible.
X = wine.data
y = wine.target
# print(f"\n--- X ---\n {X}")
# print(f"\n--- y ---\n {y}")

# Diviser les tableaux ou matrices en sous-ensembles aléatoires de formation et de test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
print(f"\n--- X_train len = {len(X_train)} Size = {math.ceil(len(X_train)/len(X)*100)}%  ---\n {X_train}")
print(f"\n--- X_test len = {len(X_test)}  Size = {math.ceil(len(X_test)/len(X)*100)}%   ---\n {X_test}")
print(f"\n--- y_train  len = {len(y_train)}  Size = {math.ceil(len(y_train)/len(y)*100)}%  ---\n {y_train}")
print(f"\n--- y_test  len = {len(y_test)}  Size = {math.ceil(len(y_test)/len(y)*100)}%  ---\n {y_test}")

# Entraîner (le modèle) un algorithme approprié (3)
# Sélectionnez un algorithme approprié pour prédire les variétés de vin. Entraînez l'algorithme.
# Utilisez le classificateur à arbre de décision comme modèle d'apprentissage automatique pour ajuster les données.
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)


# Tester l'algorithme sur les données de test. (4)
# Calculez au moins une mesure de l'exactitude de la prédiction.

# Faites des prédictions

# Illustrez votre résultat (5)
# Illustrez graphiquement le nombre de vins de chaque classe qui ont été correctement prédits. 

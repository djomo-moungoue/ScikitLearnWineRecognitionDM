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


# 1- Charger/importer les données

# Utilisez les jeux de données pour charger le jeu de données intégré sur le vin.
wine = datasets.load_wine(return_X_y=False, as_frame=False)
#print(f"\n---- sklearn wine dataset : START ---\n\n{wine}\n\n---- sklearn wine dataset : END---\n")

# 2- Divisez les données en ensembles de formation et de test
# Veillez à ce que les données soient divisées de manière aléatoire et que les classes soient équilibrées. 70% des données doivent être utilisées pour la formation.

# Créez les objets X et y pour stocker respectivement les données et la valeur cible.
X = wine.data
y = wine.target
# print(f"\n--- X ---\n {X}")
# print(f"\n--- y ---\n {y}")

# Vérifier si le jeu de donnée est équilibré ou non. Il revient à vérifier si ses classes sont équilibrées ou non.
# Déterminer le nombre de classes
wine_classes = wine.target_names
print(f"\n--- wine_classes ---\n {wine_classes}") #  ['class_0' 'class_1' 'class_2']

# Déterminer la répartition des classes
wine_description = wine.DESCR
index_from = wine_description.index(":Class Distribution:")+len(":Class Distribution:")
index_to = wine_description.index(":Creator: R.A. Fisher")
wine_classes_distribution = wine_description[index_from+1:index_to]
# print(f"\n--- wine_description ---\n {wine_description}")
print(f"\n--- wine_classes_distribution ---\n {wine_classes_distribution}") # class_0 (59), class_1 (71), class_2 (48) -> Les classes ne sont pas équilibrées


# Diviser les tableaux ou matrices en sous-ensembles aléatoires de formation et de test.
# shuffle=True : Assure la répartition des données de manière aléatoire
# : Assure l'équilibre entre les classes
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, shuffle=True)
print(f"\n--- X_train len = {len(X_train)} Size = {math.ceil(len(X_train)/len(X)*100)}%  ---")
print(f"\n--- X_test len = {len(X_test)}  Size = {math.ceil(len(X_test)/len(X)*100)}%   ---")
print(f"\n--- y_train  len = {len(y_train)}  Size = {math.ceil(len(y_train)/len(y)*100)}%  ---")
print(f"\n--- y_test  len = {len(y_test)}  Size = {math.ceil(len(y_test)/len(y)*100)}%  ---")

# 3- Entraîner (le modèle) un algorithme approprié
# Sélectionnez un algorithme approprié pour prédire les variétés de vin. Entraînez l'algorithme.
# Utilisez le classificateur à arbre de décision comme modèle d'apprentissage automatique pour ajuster les données.
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

# 4- Tester l'algorithme sur les données de test.
# Calculez au moins une mesure de l'exactitude de la prédiction.

# Faites des prédictions

# 5- Illustrez votre résultat
# Illustrez graphiquement le nombre de vins de chaque classe qui ont été correctement prédits. 

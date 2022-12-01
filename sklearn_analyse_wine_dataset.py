"""
Projet : Analyser le jeu de données Scikit-learn sur le vin (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) et utiliser les propriétés pour prédire la variété du vin.
"""
import math
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree

print("\n-------------------------------------------------------------------------------------------")
print("| Objectif du projet : Prédire les variétés de vins du jeu de données de vin scikit-learn |")
print("-------------------------------------------------------------------------------------------\n")


# 1- Charger/importer les données

# Utilisez les jeux de données pour charger le jeu de données intégré sur le vin.
wine_bunch = datasets.load_wine()
wine_bunch_df = datasets.load_wine(return_X_y=False, as_frame=True)
wine_data_target = datasets.load_wine(return_X_y=True, as_frame=False) -> X = wine_bunch.data AttributeError: 'tuple' object has no attribute 'data'
wine_data_target_df = datasets.load_wine(return_X_y=True, as_frame=True) -> X = wine_bunch.data AttributeError: 'tuple' object has no attribute 'data'


#print(f"\n---- sklearn wine dataset : START ---\n\n{wine}\n\n---- sklearn wine dataset : END---\n")

# 2- Divisez les données en ensembles de formation et de test
# Veillez à ce que les données soient divisées de manière aléatoire et que les classes soient équilibrées. 70% des données doivent être utilisées pour la formation.

# Créez les objets X et y pour stocker respectivement les données et la valeur cible.
X = wine_bunch.data
y = wine_bunch.target
# print(f"\n--- X ---\n {X}")
# print(f"\n--- y ---\n {y}")

# Vérifier si le jeu de donnée est équilibré ou non. Il revient à vérifier si ses classes sont équilibrées ou non.
# Déterminer le nombre de classes
wine_bunch_classes = wine_bunch.target_names
print(f"\n--- wine_classes ---\n {wine_bunch_classes}") #  ['class_0' 'class_1' 'class_2']

# Déterminer la répartition des classes
wine_bunch_description = wine_bunch.DESCR
index_from = wine_bunch_description.index(":Class Distribution:")+len(":Class Distribution:")
index_to = wine_bunch_description.index(":Creator: R.A. Fisher")
wine_bunch_classes_distribution = wine_bunch_description[index_from+1:index_to]
# print(f"\n--- wine_description ---\n {wine_description}")
print(f"\n--- wine_bunch_classes_distribution ---\n {wine_bunch_classes_distribution}") # class_0 (59), class_1 (71), class_2 (48) -> Les classes ne sont pas équilibrées


# Diviser les tableaux ou matrices en sous-ensembles aléatoires de formation et de test.
# shuffle=True : Assure la répartition des données de manière aléatoire
# SVC(class_weight='balanced', probability=True) : Assure l'équilibre entre les classes (https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/)
# - class_weight : dict ou 'balanced', default=None : Fixe le paramètre C de la classe i à class_weight[i]*C pour le SVC. S'il n'est pas donné, toutes les classes sont supposées avoir un poids de un. 
# Le mode "équilibré" utilise les valeurs de y pour ajuster automatiquement les poids de manière inversement proportionnelle aux fréquences des classes dans les données d'entrée comme n_échantillons / (n_classes * np.bincount(y)).
# - random_state : int, RandomState instance ou None, default=None : Contrôle la génération de nombres pseudo-aléatoires pour mélanger les données afin d'estimer les probabilités. Ignoré lorsque la probabilité est False. 
# Passez un int pour une sortie reproductible sur plusieurs appels de fonction. Voir le glossaire <random_state>.
svc_balanced_model = SVC(class_weight='balanced', probability=True)
print(f"\n--- svc_balanced_model ---\n {str(svc_balanced_model)}")

# X : 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=42, shuffle=True)
print(f"\n--- X_train len = {len(X_train)} Size = {math.ceil(len(X_train)/len(X)*100)}%  ---")
print(f"\n--- X_test len = {len(X_test)}  Size = {math.ceil(len(X_test)/len(X)*100)}%   ---")
print(f"\n--- y_train  len = {len(y_train)}  Size = {math.ceil(len(y_train)/len(y)*100)}%  ---")
print(f"\n--- y_test  len = {len(y_test)}  Size = {math.ceil(len(y_test)/len(y)*100)}%  ---")

# 3- Entraîner (le modèle) un algorithme approprié
# Sélectionnez un algorithme approprié pour prédire les variétés de vin. Entraînez l'algorithme.
# Utilisez le classificateur à arbre de décision comme modèle d'apprentissage automatique pour ajuster les données.
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

# svc_balanced_model.fit(X_train, y_train)

# svc_balanced_predict = svc_balanced_model.predict(X_test)# check performance
# print('ROCAUC score:', metrics.roc_auc_score(y_test, svc_balanced_predict))
#  print('Accuracy score:', metrics.accuracy_score(y_test, svc_balanced_predict))
# print('F1 score:', metrics.f1_score(y_test, svc_balanced_predict))

# 4- Tester l'algorithme sur les données de test.
# Calculez au moins une mesure de l'exactitude de la prédiction.
y_val = model.predict(X_test)
print(f"{metrics.classification_report(y_test, y_val)}")

# 5- Illustrez votre résultat
# Illustrez graphiquement le nombre de vins de chaque classe qui ont été correctement prédits. 
plt.bar(y_test, y_val)

plt.title("Nombre de vins de chaque classe qui ont été correctement prédits")

plt.ylabel('Y-Axis')
plt.xlabel('X-Axis')

plt.legend(["Wine Dataset Classes"])

plt.savefig("ProjectsZa/MachineLearningZa/charts/wine_dataset_barchart.jpg")

plt.show()

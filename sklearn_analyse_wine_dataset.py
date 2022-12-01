"""
Projet : Analyser le jeu de données Scikit-learn sur le vin (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) et utiliser les propriétés pour prédire la variété du vin.
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree

print("\n-------------------------------------------------------------------------------------------")
print("| Objectif du projet : Prédire les variétés de vins du jeu de données de vin scikit-learn |")
print("-------------------------------------------------------------------------------------------\n")


print(f"\n1- Charger/importer les données - DONE")


# Utilisez les jeux de données pour charger le jeu de données intégré sur le vin.
wine_bunch = datasets.load_wine() # -> dict
wine_bunch_df = datasets.load_wine(as_frame=True) # -> dict
wine_data_target = datasets.load_wine(return_X_y=True) # -> tuple
wine_data_target_df = datasets.load_wine(return_X_y=True, as_frame=True) # -> tuple

# print(f"\n---- sklearn wine_bunch dataset : START ---\n\n{wine_bunch}\n\n---- sklearn wine_bunch dataset : END---\n")
# print(f"\n---- sklearn wine_bunch_df dataset : START ---\n\n{wine_bunch_df}\n\n---- sklearn wine_bunch_df dataset : END---\n")
# print(f"\n---- sklearn wine_bunch dataset keys : START ---\n\n{str(wine_bunch.keys())}\n\n---- sklearn wine_bunch dataset keys : END---\n")
print(f"\n---- sklearn wine_bunch_df dataset keys {list(wine_bunch.keys())} : START ---\n\n")
wine_bunch_classes = wine_bunch_df["target_names"]
for key, value in wine_bunch_df.items():
    if key == "target":
        series_count = pd.Series(value).value_counts()
        print(f"Répartition des classes de vin sur un total de {len(value)} instances, soit 100%")
        for i, item in enumerate(wine_bunch_classes):
            print(f"{item} : {series_count[i]}, soit {round(series_count[i]/len(value)*100)}%")
    if key == "frame":
        print(f"\n\n--- Wine Dataset {key} ---\n")
        print(value)


print(f"\n\n---- sklearn wine_bunch_df dataset keys {list(wine_bunch.keys())} : END---\n")

# print(f"\n---- sklearn wine_data_target dataset : START ---\n\n{wine_data_target}\n\n---- sklearn wine_data_target dataset : END---\n")
# print(f"\n---- sklearn wine_data_target_df dataset : START ---\n\n{wine_data_target_df}\n\n---- sklearn wine_data_target_df dataset : END---\n")

print(f"\n2- Divisez les données en ensembles de formation et de test - DONE ")
# Veillez à ce que les données soient divisées de manière aléatoire et que les classes soient équilibrées. 70% des données doivent être utilisées pour la formation.

# Créez les objets X et y pour stocker respectivement les données et la valeur cible.
X = wine_bunch.data
y = wine_bunch.target
# X_dt = wine_data_target[0]
# y_dt = wine_data_target[1]
# X_dt_df = wine_data_target_df[0]
# y_dt_df = wine_data_target_df[1]
# print(f"\n--- X ---\n {X}")
# print(f"\n--- y ---\n {y}")
# print(f"\n--- X_dt ---\n {X_dt}")
# print(f"\n--- y_dt ---\n {y_dt}")
# print(f"\n--- X_dt_df ---\n {X_dt_df}")
# print(f"\n--- y_dt_df ---\n {y_dt_df}")

# Diviser les tableaux ou matrices en sous-ensembles aléatoires de formation et de test.
# shuffle=True : Assure la répartition des données de manière aléatoire
# SVC(class_weight='balanced', probability=True) : Assure l'équilibre entre les classes (https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/)
# - class_weight : dict ou 'balanced', default=None : Fixe le paramètre C de la classe i à class_weight[i]*C pour le SVC. S'il n'est pas donné, toutes les classes sont supposées avoir un poids de un. 
# Le mode "équilibré" utilise les valeurs de y pour ajuster automatiquement les poids de manière inversement proportionnelle aux fréquences des classes dans les données d'entrée comme n_échantillons / (n_classes * np.bincount(y)).
# - random_state : int, RandomState instance ou None, default=None : Contrôle la génération de nombres pseudo-aléatoires pour mélanger les données afin d'estimer les probabilités. Ignoré lorsque la probabilité est False. 
# Passez un int pour une sortie reproductible sur plusieurs appels de fonction. Voir le glossaire <random_state>.
svc_balanced_model = SVC(class_weight='balanced', probability=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=42)
print(f"\n        X\nX_train length {len(X_train)} and size {round(len(X_train)/len(X)*100)}%")
print(f"\nX_test length {len(X_test)} and size {round(len(X_test)/len(X)*100)}%")
print(f"\n        y\ny_train length {len(y_train)} and size {round(len(y_train)/len(y)*100)}%")
print(f"\ny_test length {len(y_test)} and size {round(len(y_test)/len(y)*100)}%")

print(f"\n3- Entraîner (le modèle) un algorithme approprié : tree.DecisionTreeClassifier()")
# Sélectionnez un algorithme approprié pour prédire les variétés de vin. Entraînez l'algorithme.
# Utilisez le classificateur à arbre de décision comme modèle d'apprentissage automatique pour ajuster les données.
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

# svc_balanced_model.fit(X_train, y_train)

# svc_balanced_predict = svc_balanced_model.predict(X_test)# check performance
# print('ROCAUC score:', metrics.roc_auc_score(y_test, svc_balanced_predict))
#  print('Accuracy score:', metrics.accuracy_score(y_test, svc_balanced_predict))
# print('F1 score:', metrics.f1_score(y_test, svc_balanced_predict))

print(f"\n4- Tester l'algorithme sur les données de test")
# Calculez au moins une mesure de l'exactitude de la prédiction.
y_val = model.predict(X_test)
print(f"{metrics.classification_report(y_test, y_val)}")

print(f"\n5- Illustrez votre résultat - ONGOING ")

# Illustrez graphiquement le nombre de vins de chaque classe qui ont été correctement prédits. 
plt.bar(y_test, y_val)

plt.title("Nombre de vins de chaque classe qui ont été correctement prédits")

plt.ylabel('Y-Axis')
plt.xlabel('X-Axis')

plt.legend(["Wine Dataset Classes"])

plt.savefig("ProjectsZa/MachineLearningZa/charts/wine_dataset_barchart.jpg")

plt.show()

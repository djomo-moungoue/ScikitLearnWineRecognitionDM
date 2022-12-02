"""
Projet : Charger le jeu de données Scikit-learn sur le vin (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) et utiliser l'algorithme approprié pour prédire la classe d'un vin en fonction de sa composition.
Project: Load the Scikit-learn wine dataset (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) and use the appropriate algorithm to predict the class of a wine based on its composition.
Projekt: Laden Sie den Scikit-learn Weindatensatz (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) und verwenden Sie den entsprechenden Algorithmus, um die Klasse eines Weins auf der Grundlage seiner Zusammensetzung vorherzusagen.
"""
import matplotlib.pyplot as plt
import pandas as pd
from os import environ
from pathlib import Path
from loggerza import LoggerZa
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree

# Afin d'éviter le message d'avertissement : Warning: QT_DEVICE_PIXEL_RATIO is deprecated. ...
environ["QT_DEVICE_PIXEL_RATIO"], environ["QT_AUTO_SCREEN_SCALE_FACTOR"], environ["QT_SCREEN_SCALE_FACTORS"], environ["QT_SCALE_FACTOR"]  = "0", "1", "1", "1"
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent
LoggerZa.log("\n\n-------------------------------------------------------------------------------------------\n| Objectif du projet : Prédire les variétés de vins du jeu de données de vin scikit-learn |\n-------------------------------------------------------------------------------------------")


LoggerZa.log(f"\n1- Charger/importer les données - DONE")

wine_bunch = datasets.load_wine() # -> dict
wine_bunch_df = datasets.load_wine(as_frame=True) # -> dict
wine_data_target = datasets.load_wine(return_X_y=True) # -> tuple
wine_data_target_df = datasets.load_wine(return_X_y=True, as_frame=True) # -> tuple

# LoggerZa.log(f"\n---- sklearn wine_bunch dataset : START ---\n\n{wine_bunch}\n\n---- sklearn wine_bunch dataset : END---\n")
# LoggerZa.log(f"\n---- sklearn wine_bunch_df dataset : START ---\n\n{wine_bunch_df}\n\n---- sklearn wine_bunch_df dataset : END---\n")
# LoggerZa.log(f"\n---- sklearn wine_data_target dataset : START ---\n\n{wine_data_target}\n\n---- sklearn wine_data_target dataset : END---\n")
# LoggerZa.log(f"\n---- sklearn wine_data_target_df dataset : START ---\n\n{wine_data_target_df}\n\n---- sklearn wine_data_target_df dataset : END---\n")

LoggerZa.log(f"---- sklearn wine_bunch_df dataset keys {list(wine_bunch.keys())} : START ---")

wine_bunch_classes = wine_bunch_df["target_names"]
print(f"Type of frame : {type(wine_bunch_df.get('frame'))}")
for key, value in wine_bunch_df.items():
    if key == "target":
        series_count = pd.Series(value).value_counts()
        LoggerZa.log(f"Répartition des classes de vin sur un total de {len(value)} instances, soit 100%")
        tmp_str = ""
        for i, item in enumerate(wine_bunch_classes):
            tmp_str += f"{item} : {series_count[i]}, soit {round(series_count[i]/len(value)*100)}%\n"
        LoggerZa.log(tmp_str)
        tmp_str = ""
    if key == "frame":
        LoggerZa.log(f"--- Wine Dataset {key} ---")
        LoggerZa.log(value)
        frame_dir = PROJECT_ROOT / "datasets"
        print(f"{frame_dir} found")
        if not Path(frame_dir).exists():
            Path(frame_dir).mkdir()
        dataset_csv = frame_dir / "wine_dataset_frame.csv"
        dataset_with_index_csv = frame_dir / "wine_dataset_frame_with_index.csv"
        dataset_json = frame_dir / "wine_dataset_frame.json"
        value.to_csv(dataset_csv, index=False) # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html <-> read_csv() : Load a CSV file into a DataFrame.
        value.to_csv(dataset_with_index_csv, index_label="record_id") # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html <-> read_csv() : Load a CSV file into a DataFrame.
        value.to_json(dataset_json, orient='table') #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html <-> read_json() : Load a JSON file into a DataFrame.

LoggerZa.log(f"---- sklearn wine_bunch_df dataset keys {list(wine_bunch.keys())} : END---")

LoggerZa.log(f"\n2- Divisez les données en ensembles de formation et de test - DONE ")
# Veillez à ce que les données soient divisées de manière aléatoire et que les classes soient équilibrées. 70% des données doivent être utilisées pour la formation.

# 2.1- Créez les objets X et y pour stocker respectivement les données et la valeur cible.
# X : Matrix contenant les 177 sortes de vin faite à de 13 composants dosés différement
# y : Vecteur contenant les trois classes de vin
X = wine_bunch.data
y = wine_bunch.target
# X_dt = wine_data_target[0]
# y_dt = wine_data_target[1]
# X_dt_df = wine_data_target_df[0]
# y_dt_df = wine_data_target_df[1]
# LoggerZa.log(f"\n--- X ---\n {X}")
# LoggerZa.log(f"\n--- y ---\n {y}")
# LoggerZa.log(f"\n--- X_dt ---\n {X_dt}")
# LoggerZa.log(f"\n--- y_dt ---\n {y_dt}")
# LoggerZa.log(f"\n--- X_dt_df ---\n {X_dt_df}")
# LoggerZa.log(f"\n--- y_dt_df ---\n {y_dt_df}")

# 2.2- Diviser les tableaux ou matrices en sous-ensembles aléatoires de formation et de test.
# shuffle=True et random_state=42 : Assure la répartition des données de manière aléatoire
# SVC(class_weight='balanced', probability=True) : Assure l'équilibre entre les classes (https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/)
# - class_weight : dict ou 'balanced', default=None : Fixe le paramètre C de la classe i à class_weight[i]*C pour le SVC. S'il n'est pas donné, toutes les classes sont supposées avoir un poids de un. 
# Le mode "équilibré" utilise les valeurs de y pour ajuster automatiquement les poids de manière inversement proportionnelle aux fréquences des classes dans les données d'entrée comme n_échantillons / (n_classes * np.bincount(y)).
# - random_state : int, RandomState instance ou None, default=None : Contrôle la génération de nombres pseudo-aléatoires pour mélanger les données afin d'estimer les probabilités. Ignoré lorsque la probabilité est False. 
# Passez un int pour une sortie reproductible sur plusieurs appels de fonction. Voir le glossaire <random_state>.

svc_balanced_model = SVC(class_weight='balanced', probability=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=42)
tmp_str = ""
tmp_str += f"Sous-ensembles de formation\n"
tmp_str += f"X_train : {len(X_train)}, soit {round(len(X_train)/len(X)*100)}%\n"
tmp_str += f"y_train : {len(y_train)}, soit {round(len(y_train)/len(y)*100)}%\n"
tmp_str += f"Sous-ensemble de test\n"
tmp_str += f"X_test : {len(X_test)}, soit {round(len(X_test)/len(X)*100)}%\n"
tmp_str += f"y_test : {len(y_test)}, soit {round(len(y_test)/len(y)*100)}%"
LoggerZa.log(tmp_str)


LoggerZa.log(f"\n3- Entraîner (le modèle) un algorithme approprié : tree.DecisionTreeClassifier() - ONGOING")
# Sélectionnez un algorithme approprié pour prédire les variétés de vin. Entraînez l'algorithme.
# Utilisez le classificateur à arbre de décision comme modèle d'apprentissage automatique pour ajuster les données.
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

# svc_balanced_model.fit(X_train, y_train)

# svc_balanced_predict = svc_balanced_model.predict(X_test)# check performance
# LoggerZa.log('ROCAUC score:', metrics.roc_auc_score(y_test, svc_balanced_predict))
#  LoggerZa.log('Accuracy score:', metrics.accuracy_score(y_test, svc_balanced_predict))
# LoggerZa.log('F1 score:', metrics.f1_score(y_test, svc_balanced_predict))

LoggerZa.log(f"\n4- Tester l'algorithme sur les données de test - ONGOING")
# Calculez au moins une mesure de l'exactitude de la prédiction.
y_val = model.predict(X_test)
LoggerZa.log(f"{metrics.classification_report(y_test, y_val)}")

LoggerZa.log(f"\n5- Illustrez votre résultat - ONGOING")

# Illustrez graphiquement le nombre de vins de chaque classe qui ont été correctement prédits. 
plt.bar(y_test, y_val)

plt.title("Nombre de vins de chaque classe qui ont été correctement prédits")

plt.ylabel('Y-Axis')
plt.xlabel('X-Axis')

plt.legend(["Wine Dataset Classes"])

plt.savefig(PROJECT_ROOT / "result/wine_dataset_barchart.jpg")

# plt.show()

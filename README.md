
Projet : Charger le jeu de données [Scikit-learn sur le vin](https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) et utiliser l'algorithme approprié pour prédire la classe d'un vin en fonction de sa composition.

Project: Load the [Scikit-learn wine dataset](https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) and use the appropriate algorithm to predict the class of a wine based on its composition.


|Statut du projet|
|---|
|ToDo|Ongoing|Done|
|:---:|:---:|:---:|
| 0 | 1 | 0 |


|Table des matières|
|---|
|[Configurer l'environment de développement](#Configurer-l'environment-de-développement)|
|[&nbsp;&nbsp;&nbsp;&nbsp;Miniconda](#Miniconda)|
|[&nbsp;&nbsp;&nbsp;&nbsp;Visual Studio Code](#Visual-Studio-Code)|
|[1- Charger/importer les données](#1--Charger/importer-les-données)|
|[2- Divisez les données en ensembles de formation et de test](#2--Divisez-les-données-en-ensembles-de-formation-et-de-test)|
|[3- Entraîner (le modèle) un algorithme approprié](#3--Entraîner-(le-modèle)-un-algorithme-approprié)|
|[4- Tester l'algorithme sur les données de test](#4--Tester-l'algorithme-sur-les-données-de-test)|
|[5- Illustrez votre résultat](#5--Illustrez-votre-résultat)|

# Configurer l'environment de développement

## Miniconda

Téléchargez [miniconda3 Windows 64-bit](https://docs.conda.io/en/latest/miniconda.html#windows-installers) pour Windows, exécutez le programme d'installation et suivez les étapes. 

Dans Windows Start, ouvrir `Anaconda Prompt (miniconda3)`

Afficher les informations concernant Miniconda3

(base) Project_ROOT>
~~~
conda info
~~~

Créer un environnement de travail

(base) Project_ROOT>
~~~
conda create --name sklearn_wine_dataset
~~~

Afficher la list des environnements disponibles
- usage: conda-env-script.py [-h] {create,export,list,remove,update,config}

(base) Project_ROOT>
~~~
conda env list
~~~

Activer l'environnement de travail

(base) Project_ROOT>
~~~
conda activate sklearn_wine_dataset
~~~

Afficher les librairies installées dans l'environnement actuel

(sklearn_wine_dataset) Project_ROOT>
~~~
conda list
~~~

Afficher les librairies et sous-librairies qui seront insallées dans l'environnement de travail

(sklearn_wine_dataset) Project_ROOT>
~~~
conda install --channel conda-forge pandas matplotlib scikit-learn imbalanced-learn --dry-run
~~~

Installer les librairies requises dans l'environnement de travail

(sklearn_wine_dataset) Project_ROOT>
~~~
conda install --channel conda-forge pandas matplotlib scikit-learn imbalanced-learn --yes
~~~

## Visual Studio Code

Téléchargez [Visual Studio Code Windows 64-bit](https://code.visualstudio.com/download) pour Windows, exécutez le programme d'installation et suivez les étapes. 

Dans Windows Start, ouvrir `Visual Studio Code`

Dans Visual Studio Code,

Conigurer l'interprète Python (1)
- Ouvrir la palette de commande (Ctrl + Shift + P)
- Taper "Python : Select Interpreter", 
- Sélectionnez la commande et vous devriez obtenir une liste des interpréteurs disponibles (ceux que l'extension Python a détectés).
- Choisir "Python 3.x.x ('sklearn_wine_dataset') ~\miniconda3\envs\sklearn_wine_dataset\python.exe   Conda"

Ouvir le Terminal : (2)
- (Ctrl + Shift + ö) ou 
- A partir du ruban `Terminal`, sélectionner `New Terminal`

Copier et coller dans le Terminal (3)

(sklearn_wine_dataset) Project_ROOT>
~~~
mkdir WineDataset
cd WineDataset
~~~

(sklearn_wine_dataset) Project_ROOT>
~~~
touch winevarietiesprediction.py
code winevarietiesprediction.py
~~~

---

# 1- Charger/importer les données
Chargez les données en mémoire, puis traitez-les. (sklearn.datasets.load_wine())

# 2- Divisez les données en ensembles de formation et de test
Veillez à ce que les données soient divisées de manière aléatoire et que les classes soient équilibrées. 70% des données doivent être utilisées pour la formation.

# 3- Entraîner (le modèle) un algorithme approprié
Sélectionnez un algorithme approprié pour prédire les variétés de vin. Entraînez l'algorithme.

# 4- Tester l'algorithme sur les données de test
Calculez au moins une mesure de l'exactitude de la prédiction.

# 5- Illustrez votre résultat
Illustrez graphiquement le nombre de vins de chaque classe qui ont été correctement prédits. 

---

[Resourse utile](https://www.projectpro.io/recipes/classify-wine-using-sklearn-tree-model)

---

### Dépannage des erreurs

#### data: Any - Instance of 'tuple' has no 'data' member Pylint(E101:no-member)
~~~
X = wine.data
~~~
Ce message superflu : Ignorer le, parce qu'il est dû au fait que [pylint ne reconnait pas les attributs créés dynamiquement](http://pylint-messages.wikidot.com/messages:e1101)

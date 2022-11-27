# MachineLearningZa
Goal : Analyze the [Sklearn wine dataset](https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) and use the properties to predict the variety of the wine.

# Load/import the data (1)
Load the data into memory and then process the data. (sklearn.datasets.load_wine())

# Clean the data

# Split the data into training and test sets (2)
Please make sure that the data is divided randomly and that the classes are balanced. 70% of the data should be used for training.

# Create a model

# Train (the model) a suitable algorithm (3)
Select a suitable algorithm to predict the varieties of wine. Train the algorithm.

# Test the algorithm on the test data. (4)
Calculate at least one measure of the accuracy of the prediction.

# Make predictions

# Illustrate your result (5)
Graphically illustrate how many wines of each class were correctly predicted. 

---

# Configuration requise

## [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
Miniconda est un installateur minimal gratuit pour conda. C'est une petite version d'Anaconda qui inclut seulement conda, Python, les paquets dont ils dépendent, et un petit nombre d'autres paquets utiles, incluant pip, zlib et quelques autres. Utilisez la commande `conda install` pour installer plus de 720 paquets conda supplémentaires depuis le dépôt d'Anaconda.

[Miniconda vs Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda)

Choisissez `Miniconda` si vous :
- Cela ne vous dérange pas d'installer individuellement chacun des paquets que vous voulez utiliser.
- Vous n'avez pas le temps ou l'espace disque nécessaire pour installer plus de 1500 paquets à la fois.
- Vous voulez un accès rapide à Python et aux commandes conda et vous souhaitez trier les autres programmes plus tard.

Choisissez `Anaconda` si vous :
- Vous êtes novice en matière de conda ou de Python.
- Vous aimez la commodité d'avoir Python et plus de 1 500 paquets scientifiques automatiquement installés en une seule fois.
- Vous avez le temps et l'espace disque nécessaires - quelques minutes et 3 Go.
- Vous ne souhaitez pas installer individuellement chacun des paquets que vous voulez utiliser.
- Vous souhaitez utiliser un ensemble de paquets sélectionnés et vérifiés pour leur interopérabilité et leur facilité d'utilisation.

[Commandes `Conda` vs `pip` vs `virtualenv`](https://docs.conda.io/projects/conda/en/latest/commands.html#conda-vs-pip-vs-virtualenv-commands) : Si vous avez utilisé pip et virtualenv dans le passé, vous pouvez utiliser conda pour effectuer toutes les mêmes opérations. Pip est un gestionnaire de paquets et virtualenv est un gestionnaire d'environnement. conda est les deux.

## [Pandas](https://pandas.pydata.org/)
Pandas est une bibliothèque Python populaire pour l'analyse des données. Elle n'est pas directement liée à l'apprentissage automatique. Comme nous le savons, l'ensemble de données doit être préparé avant la formation. Dans ce cas, Pandas est très utile car il a été développé spécifiquement pour l'extraction et la préparation des données. Il fournit des structures de données de haut niveau et une grande variété d'outils pour l'analyse des données. Il fournit de nombreuses méthodes intégrées pour regrouper, combiner et filtrer les données.
- [x] `ca-certificates`(1)
- [x] [Numpy](www.numpy.org) : NumPy est une bibliothèque python très populaire pour le traitement de grands tableaux multidimensionnels et de matrices, à l'aide d'une grande collection de fonctions mathématiques de haut niveau. (*)
- [x] `openssl`(2)
- [x] `pandas`
- [x] `pip`(3)
- [x] `setuptools`(4)
- [x] `sqlite`(5)
- [x] `wheel`(6)

## [Matplotlib](https://matplotlib.org/) 
Matplotlib est une bibliothèque Python très populaire pour la visualisation de données. Comme Pandas, elle n'est pas directement liée à l'apprentissage automatique. Elle s'avère particulièrement utile lorsqu'un programmeur souhaite visualiser les modèles dans les données. Il s'agit d'une bibliothèque de traçage en 2D utilisée pour créer des graphiques et des tracés en 2D. Un module appelé pyplot facilite la tâche des programmeurs en matière de traçage, car il fournit des fonctionnalités permettant de contrôler les styles de lignes, les propriétés des polices, le formatage des axes, etc. Il fournit différents types de graphiques et de tracés pour la visualisation des données, à savoir des histogrammes, des diagrammes d'erreur, des barres de données, etc, 
- [x] (1-6)
- [x] `fonttools`
- [x] `jpeg`
- [x] `matplotlib`
- [x] (*)
- [x] [Python](https://www.python.org) : est un langage de programmation interprété, multi-paradigme et multiplateformes. Il favorise la programmation impérative structurée, fonctionnelle et orienté objet.


## [Scikit-learn](https://scikit-learn.org/stable/index.html)
Scikit-learn est l'une des bibliothèques ML les plus populaires pour les algorithmes ML classiques. Elle est construite à partir de deux bibliothèques Python de base, à savoir NumPy et SciPy. Scikit-learn prend en charge la plupart des algorithmes d'apprentissage supervisé et non supervisé. Scikit-learn peut également être utilisé pour l'exploration et l'analyse de données, ce qui en fait un outil idéal pour les débutants en ML.
- [x] (1-6)
- [x] (*)
- [x] [python](https://www.python.org) : est un langage de programmation interprété, multi-paradigme et multiplateformes. Il favorise la programmation impérative structurée, fonctionnelle et orienté objet.
- [x] `scikit-learn`
- [x] `Scipy` : SciPy est une bibliothèque très populaire parmi les amateurs d'apprentissage automatique car elle contient différents modules d'optimisation, d'algèbre linéaire, d'intégration et de statistiques. 

## [Jupyter](https://jupyter.org/)
- [x] `beautifulsoup4`
- [x] `ipython`
- [x] `json5`
- [x] `jsonschema`
- [x] `jupyter`
- [x] `jupyter_client`
- [x] `jupyter_console`
- [x] `jupyter_core`
- [x] `jupyter_server`
- [x] `jupyterlab`
- [x] `notebook`
- [x] `requests`

# Configurer l'environment de developpement

Téléchargez [miniconda3 Windows 64-bit](https://docs.conda.io/en/latest/miniconda.html#windows-installers) pour votre système d'exploitation, exécutez le programme d'installation et suivez les étapes. 

Dans Windows Start, ouvrir `Anaconda Prompt (miniconda3)`

Afficher les informations concernant Miniconda3
~~~
(base) USER_ROOT> conda info
~~~

Créer un environnement de travail
~~~
(base) USER_ROOT> conda create --name wine_dataset
~~~

Activer l'environnement de travail
~~~
(base) USER_ROOT> conda activate wine_dataset
~~~

Afficher les librairies installées dans l'environnement actuel
~~~
(wine_dataset) USER_ROOT> conda list
~~~

Installer les librairies requises dans l'environnement de travail
~~~
(wine_dataset) USER_ROOT> conda install pandas matplotlib scikit-learn jupyter
~~~

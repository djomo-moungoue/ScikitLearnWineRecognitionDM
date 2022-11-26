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

# [Préréquis](https://www.geeksforgeeks.org/best-python-libraries-for-machine-learning/)
- [x] Miniconda
- [x] Python
- [x] Pandas : Pandas est une bibliothèque Python populaire pour l'analyse des données. Elle n'est pas directement liée à l'apprentissage automatique. Comme nous le savons, l'ensemble de données doit être préparé avant la formation. Dans ce cas, Pandas est très utile car il a été développé spécifiquement pour l'extraction et la préparation des données. Il fournit des structures de données de haut niveau et une grande variété d'outils pour l'analyse des données. Il fournit de nombreuses méthodes intégrées pour regrouper, combiner et filtrer les données.
    - [x] Numpy : NumPy est une bibliothèque python très populaire pour le traitement de grands tableaux multidimensionnels et de matrices, à l'aide d'une grande collection de fonctions mathématiques de haut niveau.
- [x] Matplotlib : Matplotlib est une bibliothèque Python très populaire pour la visualisation de données. Comme Pandas, elle n'est pas directement liée à l'apprentissage automatique. Elle s'avère particulièrement utile lorsqu'un programmeur souhaite visualiser les modèles dans les données. Il s'agit d'une bibliothèque de traçage en 2D utilisée pour créer des graphiques et des tracés en 2D. Un module appelé pyplot facilite la tâche des programmeurs en matière de traçage, car il fournit des fonctionnalités permettant de contrôler les styles de lignes, les propriétés des polices, le formatage des axes, etc. Il fournit différents types de graphiques et de tracés pour la visualisation des données, à savoir des histogrammes, des diagrammes d'erreur, des barres de données, etc, 
- [x] Scikit-learn : Scikit-learn est l'une des bibliothèques ML les plus populaires pour les algorithmes ML classiques. Elle est construite à partir de deux bibliothèques Python de base, à savoir NumPy et SciPy. Scikit-learn prend en charge la plupart des algorithmes d'apprentissage supervisé et non supervisé. Scikit-learn peut également être utilisé pour l'exploration et l'analyse de données, ce qui en fait un outil idéal pour les débutants en ML.
    - [x] Scipy : SciPy est une bibliothèque très populaire parmi les amateurs d'apprentissage automatique car elle contient différents modules d'optimisation, d'algèbre linéaire, d'intégration et de statistiques. 
- [x] TensorFlow : TensorFlow est une bibliothèque open-source très populaire pour le calcul numérique à haute performance, développée par l'équipe Google Brain de Google. Comme son nom l'indique, Tensorflow est un cadre qui permet de définir et d'exécuter des calculs impliquant des tenseurs. Il peut former et exécuter des réseaux neuronaux profonds qui peuvent être utilisés pour développer plusieurs applications d'IA. TensorFlow est largement utilisé dans le domaine de la recherche et des applications d'apprentissage profond. 
    - [x] Keras : Keras est une bibliothèque d'apprentissage automatique très populaire pour Python. Il s'agit d'une API de réseaux neuronaux de haut niveau capable de fonctionner au-dessus de TensorFlow, CNTK ou Theano.
- [ ] Theano
- [ ] PyTorch

# Configurer l'environment de developpement

## Installer Miniconda : un gestionnaire de paquets Python

Téléchargez [miniconda](https://docs.conda.io/en/latest/miniconda.html) 

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

Installer Python
~~~
(wine_dataset) USER_ROOT> conda install python
~~~

Installer Pandas
~~~
(wine_dataset) USER_ROOT> conda install pandas
~~~

Installer MatPlotLib
~~~
(wine_dataset) USER_ROOT> conda install pandas
~~~

Installer Scikit-Learn
~~~
(wine_dataset) USER_ROOT> conda install scikit-learn
~~~

Installer TensorFlow
~~~
(wine_dataset) USER_ROOT> conda install tensorflow
~~~

# Ressources 
- [ ] [Numpy](www.numpy.org) : provides a multi-dimentional array.
- [ ] [Pandas](https://pandas.pydata.org/) : a data analysis library that provides a concept called dataframe. A dataframe is two-dimentional datastructrure similar to an excel spreadsheet. 
- [ ] [MatPlotLib](https://matplotlib.org/) : a two-dimentional plotting library for creating static, animated, and interactive visualizations with Python.
- [ ] [Scikit-Learn]() : the most popular machine learning library that provide all the common algorithms like decision-trees, neuronetworks and so on.
- [ ] [Jupyter Notebook](https://jupyter.org/) : is an environment for writing our code.
    - It is more convenient than editors like VS Code for machinelearning projects, because it can better display long row of data.
    - It is installed via the installation of the Anaconda distribution.

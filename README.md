Objectif : Analyser le [jeu de données Scikit-learn sur le vin](https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) et utiliser les propriétés pour prédire la variété du vin.

|Table of Contents|
|---|
|[Configuration requise](#Configuration-requise)|
|[&nbsp;&nbsp;&nbsp;&nbsp;Miniconda](#Miniconda)|
|[&nbsp;&nbsp;&nbsp;&nbsp;Pandas](#Pandas)|
|[&nbsp;&nbsp;&nbsp;&nbsp;MatPlotLib](#MatPlotLib)|
|[&nbsp;&nbsp;&nbsp;&nbsp;Scikit-Learn](#Scikit-Learn)|
|[&nbsp;&nbsp;&nbsp;&nbsp;Jupyter](#Jupyter)|
|[Configurer l'environment de developpement](#Configurer-l'environment-de-developpement)|
|[Charger/importer les données (1)](#Charger/importer-les-données-(1))|
|[Clean the data](#Clean-the-data)|
|[Divisez les données en ensembles de formation et de test (2)](#Divisez-les-données-en-ensembles-de-formation-et-de-test-(2))|
|[Créer un modèle](#Créer-un-modèle)|
|[Entraîner (le modèle) un algorithme approprié (3)](#Entraîner-(le-modèle)-un-algorithme-approprié-(3))|
|[Tester l'algorithme sur les données de test (4)](#Tester-l'algorithme-sur-les-données-de-test-(4))|
|[Faites des prédictions](#Faites-des-prédictions)|
|[Illustrez votre résultat (5)](#Illustrez-votre-résultat-(5))|

# Configuration requise

## [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
Miniconda est un installateur minimal gratuit pour conda. C'est une petite version d'Anaconda qui inclut seulement conda, Python, les paquets dont ils dépendent, et un petit nombre d'autres paquets utiles, incluant pip, zlib et quelques autres. Utilisez la commande `conda install` pour installer plus de 720 paquets conda supplémentaires depuis le dépôt d'Anaconda.

### [Miniconda vs Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda)

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

### [Commandes `Conda` vs `pip` vs `virtualenv`](https://docs.conda.io/projects/conda/en/latest/commands.html#conda-vs-pip-vs-virtualenv-commands) : Si vous avez utilisé pip et virtualenv dans le passé, vous pouvez utiliser conda pour effectuer toutes les mêmes opérations. Pip est un gestionnaire de paquets et virtualenv est un gestionnaire d'environnement. conda est les deux.

### Librairies installées dans l'environnement (base)
- [x] `ca-certificates`(1)
- [x] `conda`
- [x] `conda-package-handling`
- [x] `cryptography`
- [x] `openssl`(2)
- [x] `pip`(3)
- [x] [python](https://www.python.org) : est un langage de programmation interprété, multi-paradigme et multiplateformes. Il favorise la programmation impérative structurée, fonctionnelle et orienté objet. (*)
- [x] `requests` (**)
- [x] `setuptools`(4)
- [x] `sqlite`(5)
- [x] `wheel`(6)

## [Pandas](https://pandas.pydata.org/)
Pandas est une bibliothèque Python populaire pour l'analyse des données. Elle n'est pas directement liée à l'apprentissage automatique. Comme nous le savons, l'ensemble de données doit être préparé avant la formation. Dans ce cas, Pandas est très utile car il a été développé spécifiquement pour l'extraction et la préparation des données. Il fournit des structures de données de haut niveau et une grande variété d'outils pour l'analyse des données. Il fournit de nombreuses méthodes intégrées pour regrouper, combiner et filtrer les données.
- [x] (1-6)
- [x] [Numpy](www.numpy.org) : NumPy est une bibliothèque python très populaire pour le traitement de grands tableaux multidimensionnels et de matrices, à l'aide d'une grande collection de fonctions mathématiques de haut niveau. (***)
- [x] `pandas`


## [MatPlotLib](https://matplotlib.org/) 
Matplotlib est une bibliothèque Python très populaire pour la visualisation de données. Comme Pandas, elle n'est pas directement liée à l'apprentissage automatique. Elle s'avère particulièrement utile lorsqu'un programmeur souhaite visualiser les modèles dans les données. Il s'agit d'une bibliothèque de traçage en 2D utilisée pour créer des graphiques et des tracés en 2D. Un module appelé pyplot facilite la tâche des programmeurs en matière de traçage, car il fournit des fonctionnalités permettant de contrôler les styles de lignes, les propriétés des polices, le formatage des axes, etc. Il fournit différents types de graphiques et de tracés pour la visualisation des données, à savoir des histogrammes, des diagrammes d'erreur, des barres de données, etc, 
- [x] (1-6)
- [x] `fonttools`
- [x] `jpeg`
- [x] `matplotlib`
- [x] (***)
- [x] (*)


## [Scikit-Learn](https://scikit-learn.org/stable/index.html)
Scikit-learn est l'une des bibliothèques ML les plus populaires pour les algorithmes ML classiques. Elle est construite à partir de deux bibliothèques Python de base, à savoir NumPy et SciPy. Scikit-learn prend en charge la plupart des algorithmes d'apprentissage supervisé et non supervisé. Scikit-learn peut également être utilisé pour l'exploration et l'analyse de données, ce qui en fait un outil idéal pour les débutants en ML.
- [x] (1-6)
- [x] (*)
- [x] (***)
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
- [x] (**)

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

Démarrer Jupyter dans le navigateur web par défaut: Interactive Computing
usage: jupyter [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir] [--paths] [--json] [--debug]
               [subcommand]
The Jupyter HTML Notebook. This launches a Tornado based HTML Notebook Server that serves up an
HTML5/Javascript Notebook client.
~~~
(wine_dataset) Project_ROOT> jupyter notebook
~~~

options:
- -h, --help : show this help message and exit
- --version : show the versions of core jupyter packages and exit
- --config-dir : show Jupyter config dir
- --data-dir : show Jupyter data dir
- --runtime-dir : show Jupyter runtime dir
- --paths : show all Jupyter paths. Add --json for machine-readable format.
- --json : output paths as machine-readable json
- --debug : output debug information about paths

Available subcommands: 
- [ ] bundlerextension 
- [ ] console 
- [ ] dejavu 
- [ ] execute 
- [ ] kernel 
- [ ] kernelspec 
- [ ] lab 
- [ ] labextension 
- [ ] labhub 
- [ ] migrate
- [ ] nbclassic 
- [ ] nbconvert 
- [ ] nbextension 
- [x] notebook
    ~~~
    jupyter notebook                       # start the notebook
    jupyter notebook --certfile=mycert.pem # use SSL/TLS certificate
    jupyter notebook password              # enter a password to protect the server
    ~~~

    Options :
    - --no-browser : Don't open the notebook in a browser after startup. Equivalent to: [--NotebookApp.open_browser=False]
    - --no-mathjax : Disable MathJax. MathJax is the javascript library Jupyter uses to render math/LaTeX. It is very large, so you may want to disable it if you have a slow internet connection, or for offline use of the notebook.
    - --log-level=<Enum> : Set the log level by value or name. Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']. Default: 30
    - --ip=<Unicode> : The IP address the notebook server will listen on. Default: 'localhost'
    - --port=<Int> : The port the notebook server will listen on (env: JUPYTER_PORT). Default: 8888
    - --keyfile=<Unicode> : The full path to a private key file for usage with SSL/TLS. Default: ''
    - --certfile=<Unicode> : The full path to an SSL/TLS certificate file. Default: ''
    - --client-ca=<Unicode> : The full path to a certificate authority certificate for SSL/TLS client
    authentication. Default: ''
    - --notebook-dir=<Unicode> : The directory to use for notebooks and kernels. Default: ''
    - --browser=<Unicode> : Specify what command to use to invoke a web browser when opening the notebook. If not specified, the default browser will be determined by the `webbrowser` standard library module, which allows setting of the BROWSER environment variable to override it. Default: ''

    Subcommands :
    - list : List currently running notebook servers.
    - stop : Stop currently running notebook server.
    - password : Set a password for the notebook server.
- [ ] qtconsole 
- [ ] run 
- [ ] script 
- [ ] server 
- [ ] serverextension 
- [ ] troubleshoot 
- [ ] trust

Créer un Notebook et le renommer 'Wine Dataset'

---

# Charger/importer les données (1)
Chargez les données en mémoire, puis traitez-les. (sklearn.datasets.load_wine())

# Clean the data

# Divisez les données en ensembles de formation et de test (2)
Veillez à ce que les données soient divisées de manière aléatoire et que les classes soient équilibrées. 70% des données doivent être utilisées pour la formation.

# Créer un modèle

# Entraîner (le modèle) un algorithme approprié (3)
Sélectionnez un algorithme approprié pour prédire les variétés de vin. Entraînez l'algorithme.

# Tester l'algorithme sur les données de test. (4)
Calculez au moins une mesure de l'exactitude de la prédiction.

# Faites des prédictions

# Illustrez votre résultat (5)
Illustrez graphiquement le nombre de vins de chaque classe qui ont été correctement prédits. 

---

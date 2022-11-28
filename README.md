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

### [`conda` vs `pipenv` pour les analystes de données](https://towardsdatascience.com/pipenv-vs-conda-for-data-scientists-b9a372faf9d9)
Si vous avez utilisé pip et virtualenv dans le passé, vous pouvez utiliser conda pour effectuer toutes les mêmes opérations. 
- Pip est un gestionnaire de paquets. 
- virtualenv est un gestionnaire d'environnement. 
- conda et pipenv gère les paquets et l'environnement. `Conda` or `pipenv` == `pip` + `virtualenv`

|critères|conda (dans Ana-/Miniconda)|pipenv (dans PyPI)|
|--|--|--|
|Le processus de mise en marche est-il simple? (1)|1|1|
|Les paquets élémentaires sont-ils disponibles? (2)|1|1|
|La résolution des dépendances est-elle correcte? (3)|1|0|
|La gestion des versions de Python est-elle correcte? (4)|1|0|
|La spécification des dépendances est-elle reproductible? (5)|0|1|
|L'espace disque occupé est-il optimal? (6)|0|1|
|L'installation des paquets est-elle sécurisée (7)|1|0|
|Longivité : Conda/pipenv est-il là pour rester ? Quelle est sa maturité ? Qui le supporte ? (8)|||
|Y-t-il de la transparence lors de l'installation des paquets ? (9)|1|0|
|Les environnement sont-t-il automatiquement créer/actualisés (10)|0|1|

### (1) Démarrer avec conda et pipenv

Pour installer conda sous Windows il faut :
1. télécharger Miniconda via [Miniconda installer for Windows](https://conda.io/miniconda.html) ou Anaconda via [Anaconda installer for Windows](https://www.anaconda.com/download/)
2. Double-cliquez sur le fichier .exe.
3. Suivez les instructions à l'écran.
    - Si vous n'êtes pas sûr d'un paramètre, acceptez les valeurs par défaut. Vous pourrez les modifier ultérieurement.
    - Lorsque l'installation est terminée, à partir du menu Démarrer, ouvrez l'invite Anaconda.
4. Testez votre installation. Dans votre fenêtre de terminal ou dans Anaconda Prompt, exécutez la commande conda list. 
~~~
(base) User_ROOT> conda list
~~~
Une liste des paquets installés apparaît si l'installation s'est déroulée correctement.

Pour installer pipenv sous Windows il faut :
1. télécharger Python via [Python installer for Windows](https://www.python.org/downloads/)
2. Double-cliquez sur le fichier .exe.
3. Suivez les instructions à l'écran.
    - Si vous n'êtes pas sûr d'un paramètre, acceptez les valeurs par défaut. Vous pourrez les modifier ultérieurement.
    - Lorsque l'installation est terminée, à partir du menu Démarrer, ouvrez l'invite Anaconda.
4. Testez votre installation. Dans votre fenêtre de terminal, exécutez la commande python --version. 
~~~
python --version
~~~
~~~
python install pip
~~~
~~~
pip install pipenv
~~~

### (2) Les paquets élémentaires pour l'analyse des données sont disponibles dans le format approprié.
~~~
conda create --name env_ds scikit-learn sqlalchemy jupyter matplotlib networkx python=3.8
~~~
~~~
pipenv install pandas scikit-learn sqlalchemy jupyter matplotlib networkx --python 3.8
~~~


### (3) Résoudre correctement les dépendances directes et indirectes

Notez qu'il est recommandé de spécifier tous les paquets en même temps pour aider Conda à résoudre les dépendances.

Conda réussit à créer un environnement et installe pandas1.0.5 qui est la dernière version de pandas à supporter numpy1.15.3.
~~~
conda create --name env_a numpy==1.15.3 pandas python=3.7
~~~

Pipenv crée un environnement utilisant numpy1.19.1, ce qui ne correspond pas à mes spécifications. Pipenv détermine qu'il y a des conflits, est incapable de créer un Pipfile.lock.
~~~
pipenv install numpy==1.15.3 pandas --python 3.7
~~~

### (4) Gérer différentes versions de Python.

Conda traitera la distribution python comme un paquet et installera automatiquement toute version de python que vous avez directement spécifiée. De plus, lors de la création d'un nouvel environnement, Conda déterminera la meilleure version de python (si elle n'est pas spécifiée). 
~~~
conda create —-name env_a pandas
~~~

Pipenv n'installe pas nativement différentes versions de python. Il utilisera le python système (généralement stocké dans /usr/lib) ou le python de base (généralement stocké dans ~/miniconda3/bin si miniconda est installé) pour créer de nouveaux environnements.
~~~
pipenv install pandas
~~~

### (5) Assurer une construction reproductible et évolutive

Conda utilise un fichier `environment.yaml` pour spécifier les dépendances directes et indirectes. Les utilisateurs doivent procéder par essais et erreurs lors de la mise à jour de leurs environnements. 

Pipenv utilise deux fichiers pour spécifier les dépendances : 
- `Pipfile` pour les dépendances directes et 
- `Pipfile.lock` pour les dépendances directes et indirectes. 
- Créer un environnement en utilisant le `Pipfile.lock` garantit que les mêmes paquets seront installés, y compris le hash du paquet. 
- La création d'un environnement à l'aide du `Pipfile` donne la possibilité de mettre à niveau les dépendances indirectes si nécessaire.

### (6) Combien d'espace les environnements prennent-ils ? Le partage peut-il aider ?

Les environnements Python utilisés par les analystes des données ont tendance à être volumineux, en particulier les environnements Conda. Par exemple, un environnement conda avec jupyter et pandas occupe 1,7 Go, tandis qu'un environnement pipenv équivalent occupe 208 Mo. Bien que cela ne concerne pas la plupart des environnements de développement, cela peut devenir plus important en production, par exemple lors de l'utilisation de conteneurs [Plus ...](https://towardsdatascience.com/how-to-shrink-numpy-scipy-pandas-and-matplotlib-for-your-data-product-4ec8d7e86ee4)

En raison de leur taille importante, les spécialistes des données utilisent souvent un environnement conda dans plusieurs projets exploratoires, voire dans plusieurs projets de production faisant partie de la même solution [Plus ...](https://stackoverflow.com/questions/55892572/keeping-the-same-shared-virtualenvs-when-switching-from-pyenv-virtualenv-to-pip)
L'environnement conda peut être créé, activé et utilisé depuis n'importe quel endroit.

Un environnement pipenv est lié à un référentiel de projet. Une fois créé, Pipenv enregistre les pipfiles à la racine du référentiel. Les paquets installés sont sauvegardés dans ~/.local/share/.virtualenvs / par défaut, où pipenv s'assure qu'un environnement est créé par repo en créant un nouveau répertoire et en ajoutant un hash du chemin au nom (i.e. my_project-a3de50). L'utilisateur doit se rendre à la racine du dépôt du projet pour activer l'environnement, mais le shell restera activé même si vous quittez le répertoire. Il est possible de partager un environnement entre plusieurs projets en stockant les Pipfiles dans un répertoire séparé. L'utilisateur doit alors se souvenir de se rendre dans le référentiel pour activer et mettre à jour l'environnement.

### (7) L'installation des paquets est-elle sécurisée

Le canal principal d'[Anaconda](https://anaconda.org/anaconda/) est maintenu par des employés d'Anaconda et les paquets passent par un contrôle de sécurité strict avant d'être téléchargés. 

Dans le cas de pipenv qui utilise PyPI, n'importe qui peut télécharger n'importe quel paquet et des paquets malveillants ont été découverts dans le passé [voir](https://www.zdnet.com/article/twelve-malicious-python-libraries-found-and-removed-from-pypi/). 

Il en va de même pour [conda-forge](https://conda-forge.org/), bien qu'ils soient en train de développer un processus pour valider les artefacts avant qu'ils ne soient téléchargés vers le dépôt.

Les solutions de contournement sont les suivantes :
- Effectuer des contrôles de sécurité en utilisant des outils comme [x-ray](https://jfrog.com/xray/)
- N'installer que des paquets datant d'au moins un mois afin de laisser suffisamment de temps pour trouver et résoudre les problèmes.

### (8) Conda/pipenv est-il là pour rester ? Quelle est sa maturité ? Qui le supporte ?

Conda/Anaconda a été créé en 2012 par la même équipe que scipy.org, qui gère la pile scipy. Conda est un outil open source, mais le référentiel anaconda est hébergé par Anaconda Inc, une organisation à but lucratif. Si conda/anaconda ne risque pas de disparaître de sitôt, cette situation a suscité des inquiétudes quant à la possibilité qu'Anaconda Inc. commence à faire payer les utilisateurs. Ils ont récemment modifié leurs conditions générales pour faire payer les utilisateurs lourds ou commerciaux, ce qui inclut la mise en miroir du référentiel anaconda. Notez que les nouvelles conditions ne s'appliquent pas au canal conda-forge.

Pipenv a été présenté pour la première fois en 2017 par le créateur de la populaire bibliothèque requests. Pipenv n'a pas publié de nouveau code entre novembre 2018 et mai 2020, ce qui a suscité des inquiétudes quant à son avenir [Plus ...](https://medium.com/telnyx-engineering/rip-pipenv-tried-too-hard-do-what-you-need-with-pip-tools-d500edc161d4). Pipenv a maintenant été repris par de nouveaux développeurs et est mis à jour plus régulièrement avec des versions mensuelles depuis mai 2020.

### (9) Y-t-il de la transparence lors de l'installation des paquets ?

Conda résout et imprime les paquets qui seront installés avant de les installer, donnant aux utilisateurs la possibilité de poursuivre ou de reconsidérer avant de passer par la longue procédure d'installation.

Changer le nom/chemin du répertoire du projet rompt l'environnement pipenv et un nouvel environnement est automatiquement créé [voir](https://github.com/pypa/pipenv/issues/796)

### (10) L'environnement est-t-il créé / mis à jour automatiquement ?

Conda ne crée pas/met à jour automatiquement le fichier environment.yaml, contrairement à pipenv qui met à jour le Pipfile. Il est donc possible que votre environnement et votre fichier environment.yaml soient désynchronisés si vous oubliez de mettre à jour votre fichier environment.yaml.


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

## [SQLAlchemy](https://www.sqlalchemy.org)
SQLAlchemy est la boîte à outils SQL et le mappeur relationnel objet de Python qui offre aux développeurs d'applications toute la puissance et la souplesse de SQL. 

SQLAlchemy considère la base de données comme un moteur d'algèbre relationnelle, et pas seulement comme une collection de tables. Les lignes peuvent être sélectionnées non seulement à partir de tables, mais aussi à partir de jointures et d'autres instructions de sélection ; n'importe laquelle de ces unités peut être composée dans une structure plus large. Le langage d'expression de SQLAlchemy s'appuie sur ce concept depuis son origine.

## [networkx](https://networkx.org)
NetworkX est un package Python pour la création, la manipulation et l'étude de la structure, de la dynamique et des fonctions des réseaux complexes.

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

Afficher la list des environnements disponibles
- usage: conda-env-script.py [-h] {create,export,list,remove,update,config}
~~~
(base) USER_ROOT> conda env list
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

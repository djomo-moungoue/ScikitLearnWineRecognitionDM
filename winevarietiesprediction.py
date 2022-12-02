"""
Projet : Charger le jeu de données Scikit-learn sur le vin (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) et utiliser l'algorithme approprié pour prédire la classe d'un vin en fonction de sa composition.
Project: Load the Scikit-learn wine dataset (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) and use the appropriate algorithm to predict the class of a wine based on its composition.
Projekt: Laden Sie den Scikit-learn Weindatensatz (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) und verwenden Sie den entsprechenden Algorithmus, um die Klasse eines Weins auf der Grundlage seiner Zusammensetzung vorherzusagen.
"""
import numpy
from pathlib import Path
from datetime import datetime
from sklearn.datasets import load_wine
from sklearn.utils import Bunch

class WineVarietiesPrediction:
    """
    Predict, based on the measurement of its thirteen different constituents, to which type a wine belongs to. 
    There are 3 types of wine : class_0, class_2 and class_3.
    """
    CURRENT_FILE = Path(__file__).resolve()
    PROJECT_ROOT = CURRENT_FILE.parent
    print(f"[{datetime.now()}] CURRENT_FILE : {CURRENT_FILE}")
    print(f"[{datetime.now()}] PROJECT_ROOT : {PROJECT_ROOT}\n")

    def __init__(self):
        pass

    def load_wine_dataset(self, as_frame=False) -> Bunch:
        """
        Load the wine recognition dataset from sklearn and return a sklearn.utils.Bunch object
        references : 
        - https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset  
        - https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch
        """
        wine_bunch = load_wine()
        print("Wine Recognition Dataset loaded...")
        return wine_bunch

    def get_data(self, wine_bunch: Bunch) -> numpy.ndarray:
        """Returns the value of the data key of the dataset."""
        X = wine_bunch['data']
        return X

    def get_target(self, wine_bunch: Bunch) -> numpy.ndarray:
        """Returns the value of the target key of the dataset."""
        y = wine_bunch['target']
        return y

        
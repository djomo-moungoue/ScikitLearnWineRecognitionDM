"""
Projet : Charger le jeu de données Scikit-learn sur le vin (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) et utiliser l'algorithme approprié pour prédire la classe d'un vin en fonction de sa composition.
Project: Load the Scikit-learn wine dataset (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) and use the appropriate algorithm to predict the class of a wine based on its composition.
Projekt: Laden Sie den Scikit-learn Weindatensatz (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) und verwenden Sie den entsprechenden Algorithmus, um die Klasse eines Weins auf der Grundlage seiner Zusammensetzung vorherzusagen.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.utils import Bunch
from winevarietiesprediction import WineVarietiesPrediction

class TestWineVarietiesPrediction(unittest.TestCase):
    """
    Contains unittests of functions implemented in WineVarietiesPrediction to ensure 100% coverage of test cases after refactoring.
    """
    CURRENT_FILE = Path(__file__).resolve()
    PROJECT_ROOT = CURRENT_FILE.parent
    print(f"[{datetime.now()}] CURRENT_FILE : {CURRENT_FILE}")
    print(f"[{datetime.now()}] PROJECT_ROOT : {PROJECT_ROOT}\n")

    def setUp(self):
        pass

    def test_if_passes(self):
        """1 == 1 should be True"""
        first = 1
        second = 1
        msg = "1 == 1 should be True"
        self.assertEqual(first, second, msg)

# 1- Charger/importer les données](#1--Charger/importer-les-données)

    def test_if_load_dataset_returns_a_wine_bunch_object(self):
        """second should be a Bunch object."""
        first = Bunch
        second = type(WineVarietiesPrediction().load_wine_dataset())
        msg = "second should be a Bunch object."
        self.assertEqual(first, second, msg)

    def test_if_load_dataset_returns_178_instances(self):
        """second should return 178."""
        first = 178
        wine_bunch = WineVarietiesPrediction().load_wine_dataset()
        second = len(wine_bunch["target"])
        msg = "second should return 178."
        self.assertEqual(first, second, msg)

    def test_if_load_dataset_returns_a_target_of_3_classes(self):
        """second should return 3."""
        first = 3
        wine_bunch = WineVarietiesPrediction().load_wine_dataset()
        second = len(set(wine_bunch["target"]))
        msg = "second should return 3."
        self.assertEqual(first, second, msg)

    def test_if_get_data_returns_a_(self):
        """second should return a """
        first = None
        wine_bunch = WineVarietiesPrediction().load_wine_dataset()
        data = WineVarietiesPrediction().get_data(wine_bunch)
        second = type(data)
        msg = "second should return a 178x13 data matrix."
        self.assertEqual(first, second, msg)

    def test_if_get_target_returns_a_tuple(self):
        """second should return a tuple"""
        first = tuple
        wine_bunch = WineVarietiesPrediction().load_wine_dataset()
        target = WineVarietiesPrediction().get_target(wine_bunch)
        second = type(target)
        msg = "second should return a 178 target vector."
        self.assertEqual(first, second, msg)

    def test_if_get_data_returns_a_178x13_data_matrix(self):
        """second should return a 178x13 data matrix."""
        first = (178,13)
        wine_bunch = WineVarietiesPrediction().load_wine_dataset()
        data = WineVarietiesPrediction().get_data(wine_bunch)
        second = np.shape(data)
        msg = "second should return a 178x13 data matrix."
        self.assertEqual(first, second, msg)

    def test_if_get_target_returns_a_178_target_vector(self):
        """second should return a 178 target vector."""
        first = 178
        wine_bunch = WineVarietiesPrediction().load_wine_dataset()
        target = WineVarietiesPrediction().get_target(wine_bunch)
        second = len(target)
        msg = "second should return a 178 target vector."
        self.assertEqual(first, second, msg)



# 2- Divisez les données en ensembles de formation et de test](#2--Divisez-les-données-en-ensembles-de-formation-et-de-test)


# 3- Entraîner (le modèle) un algorithme approprié](#3--Entraîner-(le-modèle)-un-algorithme-approprié)


# 4- Tester l'algorithme sur les données de test](#4--Tester-l'algorithme-sur-les-données-de-test)


# 5- Illustrez votre résultat](#5--Illustrez-votre-résultat)
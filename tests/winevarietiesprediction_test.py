"""
Projet : Charger le jeu de données Scikit-learn sur le vin (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) et utiliser l'algorithme approprié pour prédire la classe d'un vin en fonction de sa composition.
Project: Load the Scikit-learn wine dataset (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) and use the appropriate algorithm to predict the class of a wine based on its composition.
Projekt: Laden Sie den Scikit-learn Weindatensatz (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) und verwenden Sie den entsprechenden Algorithmus, um die Klasse eines Weins auf der Grundlage seiner Zusammensetzung vorherzusagen.
"""

import unittest
import pandas as pd
import numpy as np
import array
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
        second = type(WineVarietiesPrediction().load_wine_dataset())

        first = Bunch
        msg = "second should be a Bunch object."
        self.assertEqual(first, second, msg)

    def test_if_load_dataset_returns_178_instances(self):
        """second should return 178."""
        wine_bunch = WineVarietiesPrediction().load_wine_dataset()

        first = 178
        second = len(wine_bunch["target"])
        msg = "second should return 178."
        self.assertEqual(first, second, msg)

    def test_if_load_dataset_returns_a_target_of_3_classes(self):
        """second should return 3."""
        wine_bunch = WineVarietiesPrediction().load_wine_dataset()

        first = 3
        second = len(set(wine_bunch["target"]))
        msg = "second should return 3."
        self.assertEqual(first, second, msg)

    def test_if_get_data_returns_a_ndarray(self):
        """second should return a ndarray"""
        wine_variety_prediction = WineVarietiesPrediction()
        wine_bunch = wine_variety_prediction.load_wine_dataset()
        data = wine_variety_prediction.get_data(wine_bunch)

        first = np.ndarray
        second = type(data)
        msg = "second should return a ndarray"
        self.assertEqual(first, second, msg)

    def test_if_get_target_returns_a_ndarray(self):
        """second should return a ndarray"""
        wine_variety_prediction = WineVarietiesPrediction()
        wine_bunch = wine_variety_prediction.load_wine_dataset()
        target = wine_variety_prediction.get_target(wine_bunch)

        first = np.ndarray
        second = type(target)
        msg = "second should return a ndarray."
        self.assertEqual(first, second, msg)

    def test_if_get_data_returns_a_178x13_data_ndarray(self):
        """second should return a data ndarray of 178x13 elements."""
        wine_variety_prediction = WineVarietiesPrediction()
        wine_bunch = wine_variety_prediction.load_wine_dataset()
        data = wine_variety_prediction.get_data(wine_bunch)

        first = (178,13)
        second = np.shape(data)
        msg = "second should return a data ndarray of 178x13 elements."
        self.assertEqual(first, second, msg)

    def test_if_get_target_returns_a_178_target_ndarray(self):
        """second should return a target ndarray of 178 elements."""
        wine_variety_prediction = WineVarietiesPrediction()
        wine_bunch = wine_variety_prediction.load_wine_dataset()
        target = wine_variety_prediction.get_target(wine_bunch)

        first = 178
        second = len(target)
        msg = "second should return a target ndarray of 178 elements."
        self.assertEqual(first, second, msg)

    def test_if_split_data_returns_a_list(self):
        """second should return a list."""
        wine_variety_prediction = WineVarietiesPrediction()
        wine_bunch = wine_variety_prediction.load_wine_dataset()
        data = wine_variety_prediction.get_data(wine_bunch)
        target = wine_variety_prediction.get_target(wine_bunch)

        first = list
        second = type(wine_variety_prediction.split_input(data, target))
        msg = "second should return a list."
        self.assertEqual(first, second, msg)

    def test_if_split_data_returns_a_list_2_time_bigger(self):
        """second should return a list containing twice the number of elements in target."""
        wine_variety_prediction = WineVarietiesPrediction()
        wine_bunch = wine_variety_prediction.load_wine_dataset()
        data = wine_variety_prediction.get_data(wine_bunch)
        target = wine_variety_prediction.get_target(wine_bunch)

        first = len(target)*2
        second = len(wine_variety_prediction.split_input(data, target))
        msg = "second should return a list containing twice the number of elements in target."
        self.assertEqual(first, second, msg)



# 2- Divisez les données en ensembles de formation et de test](#2--Divisez-les-données-en-ensembles-de-formation-et-de-test)


# 3- Entraîner (le modèle) un algorithme approprié](#3--Entraîner-(le-modèle)-un-algorithme-approprié)


# 4- Tester l'algorithme sur les données de test](#4--Tester-l'algorithme-sur-les-données-de-test)


# 5- Illustrez votre résultat](#5--Illustrez-votre-résultat)
"""
Projet : Charger le jeu de données Scikit-learn sur le vin (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) et utiliser l'algorithme approprié pour prédire la classe d'un vin en fonction de sa composition.
Project: Load the Scikit-learn wine dataset (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) and use the appropriate algorithm to predict the class of a wine based on its composition.
Projekt: Laden Sie den Scikit-learn Weindatensatz (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) und verwenden Sie den entsprechenden Algorithmus, um die Klasse eines Weins auf der Grundlage seiner Zusammensetzung vorherzusagen.
"""

import unittest
from sklearn.utils import Bunch
from MachineLearningZa.winevarietiesprediction import WineVarietiesPrediction

class TestWineVarietiesPrediction(unittest.TestCase):

    def setUp(self):
        pass

    def test_if_passes(self):
        first = 1
        second = 1
        msg = "1 == 1 should be True"
        self.assertEqual(first, second, msg)

    def test_if_fail(self):
        first = 1
        second = 0
        msg = "1 == 0 should be False"
        self.assertEqual(first, second, msg)

# 1- Charger/importer les données](#1--Charger/importer-les-données)

    def test_if_load_dataset_returns_a_wine_bunch_object(self):
        first = Bunch
        second = type(WineVarietiesPrediction.load_wine_dataset())
        msg = "second should be a Bunch object."
        self.assertEqual(first, second, msg)



# 2- Divisez les données en ensembles de formation et de test](#2--Divisez-les-données-en-ensembles-de-formation-et-de-test)


# 3- Entraîner (le modèle) un algorithme approprié](#3--Entraîner-(le-modèle)-un-algorithme-approprié)


# 4- Tester l'algorithme sur les données de test](#4--Tester-l'algorithme-sur-les-données-de-test)


# 5- Illustrez votre résultat](#5--Illustrez-votre-résultat)
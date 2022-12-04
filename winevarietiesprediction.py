"""
Projet : Charger le jeu de données Scikit-learn sur le vin (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) et utiliser l'algorithme approprié pour prédire la classe d'un vin en fonction de sa composition.
Project: Load the Scikit-learn wine dataset (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) and use the appropriate algorithm to predict the class of a wine based on its composition.
Projekt: Laden Sie den Scikit-learn Weindatensatz (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) und verwenden Sie den entsprechenden Algorithmus, um die Klasse eines Weins auf der Grundlage seiner Zusammensetzung vorherzusagen.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.datasets import load_wine
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


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


    def get_data(self, wine_bunch: Bunch) -> np.ndarray:
        """Returns the value of the data key of the input dataset."""
        print("Data obtained...")
        return wine_bunch['data']


    def get_target(self, wine_bunch: Bunch) -> np.ndarray:
        """Returns the value of the target key of the input dataset."""
        print("Target obtained...")
        return wine_bunch['target']


    def split_input(self, data, target, shuffle_ : bool=True, train_size_ : float=0.70, random_state_ : int=42) -> list:
        """
        shuffle : shuffle the rows of the input before splitting. Default True
        train_size : use the given percentage of data for training purpose the rest for test purpose. Default 0.70
        random_state : Controls the shuffling applied to the data before applying the split and reproduce output across multiple function calls. Default 42
        """
        print("Input split into X_train, X_test, y_train and y_test...")
        return train_test_split(data, target, shuffle=shuffle_, train_size=train_size_, random_state=random_state_)

    def get_classes_distritution(self, train_test : list) -> dict:
        """Obtain a list of X_train, X_test, y_train, y_test data and determine the distribution of classes. Helpful to check how balanced classes are."""
        y_train_count = len(train_test[2])
        y_test_count = len(train_test[3])
        classes_distribution = {}
        for i in range(len(pd.Series(y_train).value_counts())):
            classes_distribution[f'class_{i}_train (%)'] = round(pd.Series(train_test[2]).value_counts()[i] / y_train_count*100)
        for i in range(len(pd.Series(y_test).value_counts())):
            classes_distribution[f'class_{i}_test (%)'] = round(pd.Series(train_test[3]).value_counts()[i] / y_test_count*100)
        print("Distribution of classes determined...")
        return classes_distribution
    
    def balance_classes_distribution(self, target : list, data : list) -> tuple:
        """
        Obtain a list of X_train, X_test, y_train, y_test data and balance the distribution of classes.
        
        Discussion about methods to balance imballanced classes:
        1. Collect more data -> not used, because it not possible since we are not the author of the dataset.
        2. Change the performance metric -> not used, because we don't know if it is appropriate or not.
        3. Re-sample (under- oder over-sample) the dataset -> not used, because it could be tedious since it is manual. Manually, either delete rows from the input dataset or duplicate rows from the input dataset.
        4.* Generate synthetic samples -> appriopriate, because we can import the module imbalanced-learn (https://github.com/scikit-learn-contrib/imbalanced-learn) from sklearn to, automatically, re-sample the classes of our dataset.
        5.* Try different decison tree algorithms like C4.5, C5.0, CART, and Random Forest -> appropriate, because decision tree algorithms generally perform well on imbalanced datasets. (https://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/)
        6. Try penalized models -> not used, because it is complex to set the penalty matrix.
        7. Try to detect anomaly or change in the dataset -> not used, we don't have the required skills.
        Reference : https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
        """
        smote = SMOTE(random_state=42)
        print("Distribution of classes balanced...")
        X_smote, y_smote = smote.fit_resample(data, target)  
        return (X_smote, y_smote) 


if __name__ == "__main__":
    wine_variety_prediction = WineVarietiesPrediction()
    wine_bunch = wine_variety_prediction.load_wine_dataset()
    X = wine_variety_prediction.get_data(wine_bunch), 
    y = wine_variety_prediction.get_target(wine_bunch)
    print(type(wine_variety_prediction.balance_classes_distribution(X, y)))
    X_smote, y_smote = wine_variety_prediction.balance_classes_distribution(X, y)
    X_train, X_test, y_train, y_test = wine_variety_prediction.split_input(X, y)
    print(f"Initial distribution of classes : {wine_variety_prediction.get_classes_distritution([X_train, X_test, y_train, y_test])}")
    X_smote_train, X_smote_test, y_smote_train, y_smote_test = wine_variety_prediction.split_input(X_smote, y_smote)
    print(f"Balanced distribution of classes : {wine_variety_prediction.get_classes_distritution([X_smote_train, X_smote_test, y_smote_train, y_smote_test])}")

        
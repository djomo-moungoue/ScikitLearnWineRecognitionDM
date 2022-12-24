"""
Projet : Charger le jeu de données Scikit-learn sur le vin (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) et utiliser l'algorithme approprié pour prédire la classe d'un vin en fonction de sa composition.
Project: Load the Scikit-learn wine dataset (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) and use the appropriate algorithm to predict the class of a wine based on its composition.
Projekt: Laden Sie den Scikit-learn Weindatensatz (https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_wine.html) und verwenden Sie den entsprechenden Algorithmus, um die Klasse eines Weins auf der Grundlage seiner Zusammensetzung vorherzusagen.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from datetime import datetime
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.utils import Bunch
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split


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


    def load_wine_dataset(self, as_frame_=False) -> Bunch:
        """
        Load the wine recognition dataset from sklearn and return a sklearn.utils.Bunch object
        references : 
        - https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset  
        - https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch
        """
        wine_bunch = load_wine(as_frame=as_frame_)
        print("Wine Recognition Dataset loaded...")
        return wine_bunch


    def get_data(self, wine_bunch_: Bunch) -> np.ndarray:
        """Returns the value of the data key of the input dataset."""
        print("Data obtained...")
        return wine_bunch_['data']


    def get_target(self, wine_bunch_: Bunch) -> np.ndarray:
        """Returns the value of the target key of the input dataset."""
        print("Target obtained...")
        return wine_bunch_['target']


    def split_input(self, data, target, shuffle_ : bool=True, train_size_ : float=0.70, random_state_ : int=42) -> list:
        """
        shuffle : shuffle the rows of the input before splitting. Default True
        train_size : use the given percentage of data for training purpose the rest for test purpose. Default 0.70
        random_state : Controls the shuffling applied to the data before applying the split and reproduce output across multiple function calls. Default 42
        """
        print("Input split into X_train, X_test, y_train and y_test...")
        return train_test_split(data, target, shuffle=shuffle_, train_size=train_size_, random_state=random_state_)

    def get_classes_distritution(self, target : list, train_test : list) -> dict:
        """Obtain a list of X_train, X_test, y_train, y_test data and determine the distribution of classes. Helpful to check how balanced classes are."""
        target_count = len(target)
        y_train_count = len(train_test[2])
        y_test_count = len(train_test[3])
        classes_distribution = {}
        for i in range(len(pd.Series(target).value_counts())):
            classes_distribution[f'class_{i} : {Counter(target)[i]}/{target_count}'] = f"({round(Counter(target)[i] / target_count*100)}%)"
        for i in range(len(pd.Series(train_test[2]).value_counts())):
            classes_distribution[f'class_{i}_train : {Counter(train_test[2])[i]}/{y_train_count}'] = f"({round(Counter(train_test[2])[i] / y_train_count*100)}%)"
        for i in range(len(pd.Series(train_test[3]).value_counts())):
            classes_distribution[f'class_{i}_test : {Counter(train_test[3])[i]}/{y_test_count}'] = f"({round(Counter(train_test[3])[i] / y_test_count*100)}%)"
        print("Distribution of classes determined...")
        return classes_distribution
    
    def balance_classes_distribution(self, data_target : list) -> tuple:
        """
        Obtain a list of X_train, X_test, y_train, y_test data and balance the distribution of classes.
        
        Discussion about methods to balance imballanced classes:
        1. Collect more data -> not used, because it not possible since we are not the author of the dataset.
        2. Change the performance metric -> not used, because we don't know if it is appropriate or not.
        3. Re-sample (under- oder over-sample) the dataset -> not used, because it could be tedious since it is manual. Manually, either delete rows from the input dataset or duplicate rows from the input dataset.
        4.* Generate synthetic samples -> appriopriate, because we can use the class SMOTE (Synthetic Minority Oversampling Technique)
        from the module imblearn to automatically re-sample the classes of our dataset by over-sampling underrepresented classes.
            used sources :
            - [SMOTE for Imbalanced Classification with Python](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)
            - [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/pdf/1106.1813.pdf)
            - [imblearn : imbalanced-learn library]((https://github.com/scikit-learn-contrib/imbalanced-learn))
        5.* Try different decison tree algorithms like C4.5, C5.0, CART, and Random Forest -> appropriate, because decision tree algorithms generally perform well on imbalanced datasets. (https://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/)
        6. Try penalized models -> not used, because it is complex to set the penalty matrix.
        7. Try to detect anomaly or change in the dataset -> not used, we don't have the required skills.
        
        Source : https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
        """
        smote = SMOTE()
        # lab = LabelEncoder()
        # target_transformed = lab.fit_transform(target)
        X_smote, y_smote = smote.fit_resample(data_target[0], data_target[1])
        # print(f"\n-----------X_smote-------\n{X_smote}\n-----------y_smote-------\n{y_smote}")
        print("Distribution of classes balanced...")
        return (X_smote, y_smote)

    def model_the_data(self, X_train, X_test, y_train, y_test, algo='LinearRegression') -> dict:
        """
        algo : LinearRegression (default), GaussianNB, KNeighborsClassifier, DecisionTreeClassifier, SVC
        Which algorithm is best for multiclass text classification? [towardsdatascience]
        1. Linear Support Vector Machine (SVM) is widely regarded as one of the best text classification algorithms.
        2. Naive Bayes is another muticlass text classification algorithm.
        3. The Logistic Regression algorithm can be easily generalized to multiple classes.

        What is a good accuracy for multiclass classification? [oracle]
        The prevailing metrics for evaluating a multiclass classification model are Accuracy :
        - The proportion of predictions that were correct. 
        - It is generally converted to a percentage where 100% is a perfect classifier. 
        - For a balanced dataset, an accuracy of 100%k where k is the number of classes, is a random classifier.
        
        Which metric is best for multiclass classification? [oracle]
        Most commonly used metrics for multi-classes are :
        1. F1 score, 
        2. Average Accuracy, 
        3. Log-loss.

        Which Optimizer is best for multiclass classification? [obviously]
        Multiclass Classification Neural Network using Adam Optimizer.
    	
        Is 70% accuracy good in machine learning? [obviously]
        Good accuracy in machine learning is subjective. But in our opinion, anything greater than 70% is a great model performance. In fact, an accuracy measure of anything between 70%-90% is not only ideal, it's realistic. This is also consistent with industry standards.
        
        Sources : 
        - [oracle](https://docs.oracle.com/en-us/iaas/tools/ads-sdk/latest/user_guide/eval/Multiclass.html)
        - [obviously](https://www.obviously.ai/post/machine-learning-model-performance)
        - [towardsdatascience](https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568)
        """
        match algo:
            case 'GaussianNB':
                model = GaussianNB()
            case 'KNeighborsClassifier':
                model = KNeighborsClassifier()
            case 'DecisionTreeClassifier':
                model = DecisionTreeClassifier()
            case 'SVC':
                model = SVC()
            case _:
                model = LogisticRegression()
        model.fit(X_train, y_train)
        # make predictions
        expected = y_test
        #print(f"\n{str(model).center(100, '-')}")
        #print(f"{'Expected'.center(50, ' ')}\n{expected}")
        predicted = model.predict(X_test)
        #print(f"{'Predicted'.center(50, ' ')}\n{predicted}")
        # summarize the fit of the model
        accuracy_score_za = metrics.accuracy_score(expected, predicted)
        precision_score_za = metrics.precision_score(expected, predicted, average=None)
        recall_score_za = metrics.recall_score(expected, predicted, average=None)
        f1_score_za = metrics.f1_score(expected, predicted, average=None)
        return {'expected' : expected,'predicted' : predicted,'algo_name' : algo, 'accuracy': accuracy_score_za, 'precision': precision_score_za, 'recall': recall_score_za, 'f1': f1_score_za}
        

    def illusrate_the_result(self, model, predicted, expected) -> None:
        """
        The prevailing charts and plots for multiclass classification are :
        1. the Precision-Recall Curve, 
        2. the ROC curve, 
        3. the Lift Chart, 
        4. the Gain Chart, 
        5. and the Confusion Matrix. 

        Accuracy: The proportion of predictions that were correct. It is generally converted to a percentage where 100% is a perfect classifier. For a balanced dataset, an accuracy of  where  is the number of classes, is a random classifier. An accuracy of 0% is a perfectly wrong classifier.
        
        These are inter-related with preceding metrics, and are common across most multiclass classification literature.

        Source : [Multiclass Classification](https://docs.oracle.com/en-us/iaas/tools/ads-sdk/latest/user_guide/eval/Multiclass.html)    
        """
        correct_predictions = [0, 0, 0]
        for i in range(len(predicted)):
            if predicted[i] == expected[i]:
                correct_predictions[predicted[i]] += 1
        fig = plt.figure()
        fig.suptitle(model, fontsize=16)
        ax = fig.add_axes([0,0,1,1])
        classes = ['class 0', 'class 1', 'class 2']
        ax.bar(classes,correct_predictions)
        plt.savefig(Path('result') / model)
        plt.show()
        

if __name__ == "__main__":
    wine_variety_prediction = WineVarietiesPrediction()
    wine_bunch = wine_variety_prediction.load_wine_dataset()
    X = wine_variety_prediction.get_data(wine_bunch)
    y = wine_variety_prediction.get_target(wine_bunch)
    X_train, X_test, y_train, y_test = wine_variety_prediction.split_input(X, y)
    classes_distribution = wine_variety_prediction.get_classes_distritution(y, [X_train, X_test, y_train, y_test])
    print(f"\nImbalanced distribution of classes :\n")
    for k, v in classes_distribution.items():
        print(f"{k} {v}")
    #scores = wine_variety_prediction.model_the_data(X_smote_train, X_smote_test, y_smote_train, y_smote_test)
    X_smote, y_smote = wine_variety_prediction.balance_classes_distribution([X, y])
    X_smote_train, X_smote_test, y_smote_train, y_smote_test = wine_variety_prediction.split_input(X_smote, y_smote)
    classes_distribution = wine_variety_prediction.get_classes_distritution(y_smote, [X_smote_train, X_smote_test, y_smote_train, y_smote_test])
    print(f"\nBalanced distribution of classes :\n")
    for k, v in classes_distribution.items():
        print(f"{k} {v}")

    algo_performances = {'DecisionTreeClassifier':0, 'GaussianNB':0,'KNeighborsClassifier':0,'LinearRegression':0,'SVC':0}
    counter = 1 #1000
    while counter > 0:
        print("\nALL\n".center(50, '*'))
        accuracies = {}
        scores = wine_variety_prediction.model_the_data(X_smote_train, X_smote_test, y_smote_train, y_smote_test)
        accuracies[scores['algo_name']] = scores['accuracy']
        wine_variety_prediction.illusrate_the_result(scores['algo_name'], scores['predicted'], scores['expected'])


        scores = wine_variety_prediction.model_the_data(X_smote_train, X_smote_test, y_smote_train, y_smote_test, 'GaussianNB')
        accuracies[scores['algo_name']] = scores['accuracy']
        wine_variety_prediction.illusrate_the_result(scores['algo_name'], scores['predicted'], scores['expected'])
        
        scores = wine_variety_prediction.model_the_data(X_smote_train, X_smote_test, y_smote_train, y_smote_test, 'KNeighborsClassifier')
        accuracies[scores['algo_name']] = scores['accuracy']
        wine_variety_prediction.illusrate_the_result(scores['algo_name'], scores['predicted'], scores['expected'])
        
        scores = wine_variety_prediction.model_the_data(X_smote_train, X_smote_test, y_smote_train, y_smote_test, 'DecisionTreeClassifier')
        accuracies[scores['algo_name']] = scores['accuracy']
        wine_variety_prediction.illusrate_the_result(scores['algo_name'], scores['predicted'], scores['expected'])
        
        scores = wine_variety_prediction.model_the_data(X_smote_train, X_smote_test, y_smote_train, y_smote_test, 'SVC')
        accuracies[scores['algo_name']] = scores['accuracy']
        wine_variety_prediction.illusrate_the_result(scores['algo_name'], scores['predicted'], scores['expected'])
        
        sorted_accuracies = sorted(accuracies.items(), key= lambda kv:kv[1], reverse=True)
        #print("\n\nACCURACIES\n".center(50, '*'))
        #for k, v in dict(sorted_accuracies).items():
        #    print(f"{k} : {v}")
        algo_performances[sorted_accuracies[0][0]] += 1
        counter -= 1
    
    sorted_algo_performances = sorted(algo_performances.items(), key= lambda kv:kv[1], reverse=True)
    print("\n\n")
    print("MOST APPROPRIATE ALGO".center(100, '*'))
    for k, v in dict(sorted_algo_performances).items():
        print(f"{k} : {v}")


    



        
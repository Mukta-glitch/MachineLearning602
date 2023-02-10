#!/usr/bin/python3

"""References=
1.https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
2.https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
3.https://plotly.com/python/figure-labels/
4.https://stackoverflow.com/questions/57977245/python-how-do-i-create-a-new-variable-which-is-a-boolean-indicator-of-another-v"""

import sys
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree, metrics
from sklearn.metrics import accuracy_score, classification_report
import random
from scipy import stats
import plotly.express as plt
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as plt2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


class IrisDataset:
    def __init__(self, iris_dataset: str):
        self.iris_dataset = iris_dataset
        return


class hw_01(IrisDataset):

    # Load the iris data.
    def load(self):
        return pd.read_csv(
            self.iris_dataset,
            names=[
                'sepal length (cm)',
                'sepal width (cm)',
                'petal length (cm)',
                'petal width (cm)',
                'class'])

    # To perform statistical analysis.
    def npStats(self, data, col):
        min = np.min(data[col])
        max = np.max(data[col])
        mean = round(np.mean(data[col]), 2)
        std = round(np.std(data[col]), 2)
        q1 = np.quantile(data[col], 0.25)
        q2 = np.quantile(data[col], 0.5)
        q3 = np.quantile(data[col], 0.75)
        q4 = np.quantile(data[col], 0.1)
        return min, max, mean, std, q1, q2, q3, q4

    # To calculate and print the statistic summary in tabluar format.
    def statisticSummary(self, data):
        print("Summary Statistics: \n")
        print(
            "{:<20} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}". format(
                "Attribute",
                "Min",
                "Max",
                "Mean",
                "SD",
                "Q1(25%)",
                "Q2(50%)",
                "Q3(75%)",
                "Q4(100%)"))
        for i in data.iloc[:, :-1]:
            min, max, mean, std, q1, q2, q3, q4 = self.npStats(data, i)
            print(
                "{:<20} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}". format(
                    i,
                    min,
                    max,
                    mean,
                    std,
                    q1,
                    q2,
                    q3,
                    q4))

  # Plotting graphs according to different classes.
    def plotGraphs(self, data):
        fig = plt.scatter(
            data,
            x=data['sepal length (cm)'],
            y=data['sepal width (cm)'],
            color=data['class'])
        fig.write_html(file="Scatter.html", include_plotlyjs="cdn")
        fig.show()

        fig2 = plt.violin(
            data,
            x=data['sepal length (cm)'],
            y=data['sepal width (cm)'],
            color=data['class'])
        fig2.write_html(file="Violin.html", include_plotlyjs="cdn")
        fig2.show()

        fig3 = plt.pie(
            data,
            values='sepal length (cm)',
            names='class',
            color=data['class'])
        fig3.write_html(file="Pie-Chart.html", include_plotlyjs="cdn")
        fig3.show()

        fig4 = plt.histogram(
            data,
            x=data['petal length (cm)'],
            y=data['petal length (cm)'],
            color=data['class'],
            title="Histogram for Iris dataset")
        fig4.write_html(file="Histogram.html", include_plotlyjs="cdn")
        fig4.show()
        
        fig5 = plt.line(
            data,
            x=data['sepal length (cm)'],
            y=data['petal length (cm)'],
            color=data['class'],
            title="Line Graph for Iris dataset")
        fig5.write_html(file="Line.html", include_plotlyjs="cdn")
        fig5.show()
    # Different Models for classification.

    def classificationModels(self, data):
        le = sk.preprocessing.LabelEncoder()
        data['class'] = le.fit_transform(data['class'])
        X = data.iloc[:, :-1]
        y = data['class']
        print(X, y)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=50, test_size=0.35)
        print("Decision Tree Classifier")
        clf = tree.DecisionTreeClassifier(max_depth=5)
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        print(
            "Accuracy Score of depth {}: {}\t".format(
                5, accuracy_score(
                    y_test, predict)))
        print(classification_report(y_test, predict))
        pipe = Pipeline([('scaler', StandardScaler()),
                        ('Decision Tree', tree.DecisionTreeClassifier())])
        print(pipe.fit(X_train, y_train))
        print(pipe.score(X_test, y_test))

        print("Random Forest Classifier")
        clf = RandomForestClassifier(n_estimators=5)
        clf.fit(X_train, y_train)
        predictRandom = clf.predict(X_test)
        print("Accuracy:", metrics.accuracy_score(y_test, predictRandom))
        print(classification_report(y_test, predictRandom))
        pipe = Pipeline([('scaler', StandardScaler()),
                        ('Random Forest', RandomForestClassifier())])
        print(pipe.fit(X_train, y_train))
        print(pipe.score(X_test, y_test))
        data['IsAboveMean'] = data['sepal length (cm)'] > data['sepal length (cm)'].mean(
        )
        fig = plt.histogram(
            data,
            x=data['sepal length (cm)'],
            y=data['sepal width (cm)'],
            color=data['IsAboveMean'])
        fig.show()
        print(data)

        print("SVM-Linear")
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        print("Accuracy:", metrics.accuracy_score(y_test, prediction))
        print(classification_report(y_test, prediction))
        pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
        print(pipe.fit(X_train, y_train))
        print(pipe.score(X_test, y_test))


def main():
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/"
    hw01 = hw_01(iris_dataset=path)
    df = hw01.load()
    hw01.statisticSummary(df)
    hw01.plotGraphs(df)
    hw01.classificationModels(df)


if __name__ == "__main__":
    sys.exit(main())

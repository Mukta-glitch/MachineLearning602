#!/usr/bin/python3
# pylint: disable=consider-using-f-string
"""References=
1.https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
2.https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
3.https://plotly.com/python/figure-labels/
4.https://stackoverflow.com"""


import sys
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree, metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import plotly.express as plt

class IrisDataset:
    """Initialize Dataset"""
    def __init__(self, iris_dataset: str):
        self.iris_dataset = iris_dataset
        return


class HomeWork(IrisDataset):
    """Load the iris data."""
    def load(self):
        """load"""
        return pd.read_csv(
            self.iris_dataset,
            names=[
                'sepal length (cm)',
                'sepal width (cm)',
                'petal length (cm)',
                'petal width (cm)',
                'class'])

    # To perform statistical analysis.
    def np_stats(self, data, col):
        """Stat Analysis"""
        minimum = np.min(data[col])
        maximum = np.max(data[col])
        mean = round(np.mean(data[col]), 2)
        std = round(np.std(data[col]), 2)
        q_1 = np.quantile(data[col], 0.25)
        q_2 = np.quantile(data[col], 0.5)
        q_3 = np.quantile(data[col], 0.75)
        q_4 = np.quantile(data[col], 0.1)
        return minimum, maximum, mean, std, q_1, q_2, q_3, q_4

    # To calculate and print the statistic summary in tabluar format.
    def statistic_summary(self, data):
        """Stat Summary"""
        print("Summary Statistics: \n")
        print(
            "{:<20} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}". format(
                "Attribute",
                "Min",
                "Max",
                "Mean",
                "SD",
                "q_1(25%)",
                "q_2(50%)",
                "q_3(75%)",
                "q_4(100%)"))
        for i in data.iloc[:, :-1]:
            minimum, maximum, mean, std, q_1, q_2, q_3, q_4 = self.np_stats(data, i)
            print(
                "{:<20} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}". format(
                    i,
                    minimum,
                    maximum,
                    mean,
                    std,
                    q_1,
                    q_2,
                    q_3,
                    q_4))

  # Plotting graphs according to different classes.
    def plot_graphs(self, data):
        """Plot Graphs"""
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


    def classification_models(self, data):
        """Classify models"""
        l_classify = sk.preprocessing.LabelEncoder()
        data['class'] = l_classify.fit_transform(data['class'])
        x_data= data.iloc[:, :-1]
        y_data = data['class']
        print(x_data, y_data)
        scaler = StandardScaler()
        x_data = scaler.fit_transform(x_data)
        print(x_data, y_data)
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, random_state=50, test_size=0.35)
        print("Decision Tree Classifier")
        clf = tree.DecisionTreeClassifier(max_depth=5)
        clf.fit(x_train, y_train)
        predict = clf.predict(x_test)
        print(
            "Accuracy Score of depth {}: {}\t".format(
                5, accuracy_score(
                    y_test, predict)))
        print(classification_report(y_test, predict))
        pipe = Pipeline([('scaler', StandardScaler()),
                        ('Decision Tree', tree.DecisionTreeClassifier())])
        print(pipe.fit(x_train, y_train))
        print(pipe.score(x_test, y_test))

        print("Random Forest Classifier")
        clf = RandomForestClassifier(n_estimators=5)
        clf.fit(x_train, y_train)
        predict_random = clf.predict(x_test)
        print("Accuracy:", metrics.accuracy_score(y_test, predict_random))
        print(classification_report(y_test, predict_random))
        pipe = Pipeline([('scaler', StandardScaler()),
                        ('Random Forest', RandomForestClassifier())])
        print(pipe.fit(x_train, y_train))
        print(pipe.score(x_test, y_test))
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
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_test)
        print("Accuracy:", metrics.accuracy_score(y_test, prediction))
        print(classification_report(y_test, prediction))
        pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
        print(pipe.fit(x_train, y_train))
        print(pipe.score(x_test, y_test))


def main():
    """main"""
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    hw01 = HomeWork(iris_dataset=path)
    d_f = hw01.load()
    hw01.statistic_summary(d_f)
    hw01.plot_graphs(d_f)
    hw01.classification_models(d_f)


if __name__ == "__main__":
    sys.exit(main())


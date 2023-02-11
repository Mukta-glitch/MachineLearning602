#!/usr/bin/python3
"""Plot"""
import pandas as pd
import numpy as np
import sklearn as sk
import plotly.graph_objects as go

df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
df.columns = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)',
    'class']
summed_values = df.groupby(by="class", as_index=False).sum(numeric_only=True)

le = sk.preprocessing.LabelEncoder()
summed_values['class'] = le.fit_transform(summed_values['class'])

summed_values = summed_values.sort_values(by="class")

for i in [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
        'petal width (cm)']:
    for j in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:
        # print(i, j)
        pop, bins = np.histogram(
            df[i], bins=10, range=(
                np.min(
                    df[i]), np.max(
                    df[i])))
        # print(pop, bins)
        mean_pop = pop / np.mean(pop) * 0.7
        # print(mean_pop)
        iris = df[df['class'] == j].values[:, :-1] / len(df)
        # print("iris:\n", iris)
        m = np.array(iris)

        fig = go.Figure(
            data=go.Bar(
                x=bins * 0.5,
                y=pop,
                name=i,
                marker=dict(color="skyblue"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=bins * 0.5,
                y=mean_pop,
                yaxis="y2",
                name="Response",
                marker=dict(color="green"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bins,
                y=np.array(0.5 * len(bins)),
                yaxis="y2",
                name=i,
                mode="Scatter",
                marker=dict(color="crimson"),
            )
        )

        title = dict(text="Plot to analyze IRIS DATASET")

        fig.update_layout(
            legend=dict(orientation="v"),
            yaxis=dict(
                title=dict(text="Sepal Width"),
                side="left",
                range=[0, 50],
            ),
            yaxis2=dict(
                title=dict(text="Sepal Length"),
                side="right",
                range=[0, 2],
                overlaying="y",
                tickmode="auto",
            ),

            xaxis=dict(title=dict(text="Total Population"))

        )

        fig.show()
        
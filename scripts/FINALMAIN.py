import sys

import matplotlib as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.io as pio
import pyspark.sql.functions as F
import sqlalchemy
import statsmodels.api as sm
from plotly.subplots import make_subplots
from pyspark import StorageLevel, keyword_only
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import CountVectorizer, VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.regression import LinearRegression
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, lit, round, split, when
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sqlalchemy import create_engine

pio.renderers.default = "browser"
import itertools
import os
import warnings

from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

response_type_arr, predictor_type_arr = [], []


def categorizing_data(predictor_df, response_df):
    #     print(predictor_df, response_df)
    if len(set(response_df)) == 2:
        response_type = "Boolean"
        response_type_arr.append("Boolean")
        print(
            "\n Response column " + response_df.name + " is of type: " + response_type
        )
    else:
        response_type = "Continuous"
        response_type_arr.append("Continuous")
        print(
            "\n Response column " + response_df.name + " is of type: " + response_type
        )

    # loop through each predictor column
    for predictor_col in predictor_df.columns:
        # determine if predictor is cat/cont
        if predictor_df[predictor_col].dtypes == "int32":
            predictor_type = "Categorical"
            predictor_type_arr.append("Categorical")
    categorical_arr = predictor_df.select_dtypes(include="int32").columns.tolist()
    continous_arr = predictor_df.select_dtypes(exclude="int32").columns.tolist()
    categorical_df = predictor_df[categorical_arr]
    continous_df = predictor_df[continous_arr]
    # print(categorical_df)
    # print(continous_df)

    return (continous_df, categorical_df)


import plotly.graph_objects as go


def correlation_matrix(predictor_df, response_df):
    corr_matrix = predictor_df.corr()

    # Create a trace for the heatmap
    trace = go.Heatmap(
        x=predictor_df.columns,
        y=predictor_df.columns,
        z=corr_matrix,
        zmin=-1,
        zmax=1,
        colorscale="RdBu",
        colorbar=dict(
            title="Correlation",
            tickmode="array",
            tickvals=[-1, 0, 1],
            ticktext=["-1", "0", "1"],
        ),
        hovertemplate="Variable 1: %{y}<br>Variable 2: %{x}<br>Correlation: %{z:.2f}<extra></extra>",
    )

    # Create the layout
    layout = go.Layout(
        title={
            "text": "For Response Variable: " + response_df.name,
            "font": {"size": 18},
        },
        xaxis={"title": "", "showticklabels": True, "tickangle": 45},
        yaxis={"title": "", "showticklabels": True, "tickangle": 0},
        margin={"l": 100, "r": 100, "t": 100, "b": 100},
        height=600,
        width=800,
    )

    # Create the figure
    fig = go.Figure(data=[trace], layout=layout)

    # Show the figure
    fig.show()
    return fig


def plot_categorical_predictor_with_boolean_response(df, feature_names, y, response):

    df_plot = df[[response, feature_names]]
    fig, ax = plt.subplots()
    sns.kdeplot(data=df_plot, x=feature_names, y=response, fill=True, ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel(response)
    ax.set_title(feature_names)
    plt.show()


def plot_continuous_predictor_with_boolean_response(df, feature_names, y, response):

    fig, ax = plt.subplots()

    # Plot histograms
    for val in df[response].unique():
        ax.hist(
            df[df[response] == val][feature_names],
            bins=20,
            alpha=0.7,
            density=True,
            label=val,
        )

    # Set plot title and axis labels
    ax.set_title(
        f"{feature_names.capitalize()} continous predictor distribution by Boolean Response"
    )
    ax.set_xlabel(feature_names.capitalize())
    ax.set_ylabel("Density")

    # Add legend
    ax.legend()

    # Show plot
    plt.show()


def plot_categorical_predictor_with_continuous_response(feature, column, Y, response):
    print("plot_categorical_predictor_with_continuous_response")


# Function for plotting scatter plot
def plot_continuous_predictor_with_continuous_response(
    feature, feature_names, y, response
):
    if isinstance(feature, pd.DataFrame):
        print("feature is a DataFrame object")
        feature = feature[feature_names]
    else:
        fig, ax = plt.subplots()
        ax.scatter(feature, y, s=10, c="b")
        z = np.polyfit(feature, y, 1)
        p = np.poly1d(z)
        ax.plot(feature, p(feature), "r--")

        # Set labels and title
        ax.set_xlabel(feature_names)
        ax.set_ylabel(response)
        ax.set_title("Continuous Response by Continuous Predictor")

        # Show the plot
        plt.show()


def mean_square_error(X, predictors_names, y):
    df = []
    data1 = []
    rank, mean_diff_plot = {}, {}
    table_columns = [
        "BinCenter",
        "BinCount",
        "BinMean",
        "PopulationMean",
        "MeanSquareDiff",
    ]
    n = 15
    response_list = y.tolist()

    for idx, i in enumerate(predictors_names):
        predictor_list = X[i].tolist()
        bin_statistic = stats.binned_statistic(
            predictor_list, response_list, statistic="count", bins=n
        )
        bin_counts = bin_statistic.statistic
        bin_statistic = stats.binned_statistic(
            predictor_list, response_list, statistic="mean", bins=n
        )
        bin_means = bin_statistic.statistic
        bin_statistic = stats.binned_statistic(
            predictor_list, predictor_list, statistic="median", bins=n
        )
        bin_centers = bin_statistic.statistic

        population_mean = np.mean(response_list)
        mean_square_diff = (bin_means - population_mean) ** 2

        weighted_mean_square_diff = mean_square_diff * (bin_counts / len(X[i]))
        unweighted_msd = np.nansum(mean_square_diff) / n
        weighted_msd = np.nansum(weighted_mean_square_diff) / n
        rank[i] = weighted_msd

        table = {
            "BinCenter": bin_centers,
            "BinCount": bin_counts,
            "BinMean": bin_means,
            "PopulationMean": population_mean,
            "MeanSquareDiff": weighted_mean_square_diff,
        }
        df.append(pd.DataFrame(table, columns=table_columns))
        data1.append(
            {
                "response": y.name,
                "predictor": i,
                "Unweighted_rank": unweighted_msd,
                "Weighted_rank": weighted_msd,
            }
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=bin_centers, y=bin_counts, name="Hist", marker_color="blue")
        )
        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=bin_means,
                mode="lines",
                name="Mean Difference",
                line=dict(color="red"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=bin_means,
                mode="lines",
                name="Population Difference",
                line=dict(color="green"),
                marker=dict(symbol="circle"),
            )
        )

        fig.update_layout(
            title=f"Predictor {i}",
            xaxis_title="Bins",
            yaxis_title="Count",
            yaxis_tickcolor="blue",
            legend=dict(x=1, y=1),
            height=600,
            width=800,
            margin=dict(l=100, r=100, t=100, b=100),
        )

        fig.show()

        df2 = pd.DataFrame(data1)
        final_df = df2.sort_values(by="Weighted_rank", ascending=False)
        # Print the table
        print(table, data1)
    return rank, final_df


# Define the models and their corresponding hyperparameters
def regression_models(predictor_df, response_df, continous_df, categorical_df):
    if isinstance(predictor_df, pd.Series):
        predictor_df = predictor_df.to_frame()
    if isinstance(response_df, pd.Series):
        response_df = response_df.to_frame()

    print(response_df.columns, predictor_df.columns)
    models = {
        "Linear Regression": {"model": LinearRegression(), "param_grid": {}},
        "Support Vector Machine": {
            "model": SVR(),
            "param_grid": {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf", "poly"],
                "model__degree": [2, 3, 4],
            },
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(),
            "param_grid": {
                "model__max_depth": [10, 20, 30],
                "model__min_samples_leaf": [1, 2, 4],
            },
        },
        "Random Forest": {
            "model": RandomForestRegressor(),
            "param_grid": {
                "model__n_estimators": [50, 100, 150],
                "model__max_depth": [10, 20, 30],
                "model__min_samples_leaf": [1, 2, 4],
            },
        },
    }

    print(response_df.columns, predictor_df.columns)
    X = predictor_df
    y = response_df["home_team_wins"]

    # split predictor dataframe into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # preprocess the data
    continuous_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        [("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", continuous_transformer, continous_df.columns),
            ("categorical", categorical_transformer, categorical_df.columns),
        ]
    )

    # iterate over the models and their corresponding hyperparameters
    for name, model_params in models.items():
        print(f"Training {name}")
        model = model_params["model"]
        param_grid = model_params["param_grid"]

        # create a pipeline that includes preprocessing and the current model
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        # tune hyperparameters of the pipeline using cross-validation
        cv = GridSearchCV(pipeline, param_grid, cv=3)
        cv.fit(X_train, y_train)
        print(f"Best parameters for {name}:", cv.best_params_)
        print(f"Best score for {name}:", cv.best_score_)

        # train the pipeline on the entire dataset using the best hyperparameters
        pipeline.set_params(**cv.best_params_)
        pipeline.fit(predictor_df, response_df)

        # evaluate the pipeline on the testing set
        y_pred = pipeline.predict(X_test)
        print(f"Mean squared error for {name}:", mean_squared_error(y_test, y_pred))
        print("R2 score:", r2_score(y_test, y_pred))


# SVM gives better results as mean squared error is lowest - Mean squared error for Support Vector Machine: 0.04871777453323358

# Plotting all the maps based on data types of the columns
def plot_categorical_predictor_with_boolean_response(df, feature_names, y, response):
    df_plot = df[[response, feature_names]]
    fig = px.density_heatmap(df_plot, x=feature_names, y=response)
    fig.update_xaxes(title=column)
    fig.update_yaxes(title=response)
    fig.show()


def plot_continuous_predictor_with_boolean_response(df, feature_names, y, response):

    fig = px.histogram(
        df,
        x=feature_names,
        color=response,
        nbins=20,
        barmode="overlay",
        opacity=0.7,
        histnorm="percent",
    )
    fig.update_layout(
        title=f"{column.capitalize()} Distribution by Boolean Response",
        xaxis_title=column.capitalize(),
        yaxis_title="Density",
    )
    fig.show()


def plot_categorical_predictor_with_continuous_response(
    feature, feature_names, y, response
):
    group_labels = feature.unique()
    fig = go.Figure()
    for group in group_labels:
        fig.add_trace(
            go.Violin(
                x=feature[y == 1][feature == group],
                name=str(group) + " (1)",
                box_visible=True,
                meanline_visible=True,
            )
        )
        fig.add_trace(
            go.Violin(
                x=feature[y == 0][feature == group],
                name=str(group) + " (0)",
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig.update_layout(
        title="Continuous Response by Categorical Predictor",
        xaxis_title=feature_names,
        yaxis_title=response,
    )
    fig.show()


# Function for plotting scatter plot
def plot_continuous_predictor_with_continuous_response(
    feature, feature_names, y, response
):
    if isinstance(feature, pd.DataFrame):
        print("feature is a DataFrame object")
        feature = feature[feature_names]
    else:
        df = pd.DataFrame()
        df[feature_names] = feature
        df[response] = y
        fig = px.scatter(df, x=feature_names, y=response, trendline="ols")
        fig.update_layout(title="Continuous Response by Continuous Predictor")
        fig.show()


# Logistic regression
def logistic_regression(y, pred, column):
    model = statsmodels.api.Logit(y, pred)
    fitted_model = model.fit()
    print(f"Feature Name: {feature_names}")
    print(fitted_model.summary())
    return (
        round(fitted_model.tvalues[1], 6),
        fitted_model.pvalues[1],
        f"Variable: {feature_names}",
    )


def calculate_ranking(data, weights=None):
    # calculate the mean response for each combination of factors
    if weights is None:
        bin_means = (
            data.groupby(["factor1", "factor2"])["response"].mean().reset_index()
        )
    else:
        data["weighted_response"] = data["response"] * weights
        bin_means = (
            data.groupby(["factor1", "factor2"])["weighted_response"].sum()
            / data.groupby(["factor1", "factor2"])[weights].sum()
        )
        bin_means = bin_means.reset_index(name="response")

    # calculate the overall mean response
    if weights is None:
        pop_mean = data["response"].mean()
    else:
        pop_mean = np.average(data["response"], weights=weights)

    # calculate the difference between the bin means and the overall mean
    bin_means["diff_from_pop_mean"] = bin_means["response"] - pop_mean

    # sort the bin means in descending order of the difference
    bin_means = bin_means.sort_values(by="diff_from_pop_mean", ascending=False)

    # assign ranks to the bin means
    bin_means["rank"] = np.arange(len(bin_means)) + 1

    return bin_means


def plot_surface(bin_means):
    # create a 3D plot of the bin means
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    x = bin_means["factor1"]
    y = bin_means["factor2"]
    z = bin_means["response"]
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)

    # scale the legend so that the center is at the population mean
    pop_mean = bin_means["response"].mean()
    norm = mpl.colors.Normalize(
        vmin=bin_means["response"].min(), vmax=bin_means["response"].max()
    )
    sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, shrink=0.5, aspect=5, label="Response")
    ax.set_zlim(
        pop_mean - 2 * (pop_mean - bin_means["response"].min()),
        pop_mean + 2 * (bin_means["response"].max() - pop_mean),
    )

    # add labels and title to the plot
    ax.set_xlabel("Factor 1")
    ax.set_ylabel("Factor 2")
    ax.set_zlabel("Response")
    ax.set_title("Bin Means as 3D Surface Plot")

    plt.show()


def main():

    # create a connection to the database
    user = "root"
    password = "root"
    host = "localhost"
    db = "baseball"
    c = f"mariadb+mariadbconnector://{user}:{password}@{host}/{db}"  # pragma: allowlist secret
    query = "SELECT * FROM joined1"
    sql_engine = sqlalchemy.create_engine(c)
    df = pd.read_sql_query(query, sql_engine)

    predictor_df = df.drop(["home_team_wins"], axis=1)
    response_df = df["home_team_wins"]

    # joined_df = response_df.join(predictor_df, "team_id")
    print(response_df.head(5))
    print(predictor_df.head(5))
    # print(joined_df.head(5))
    print(len(response_df))
    print(len(predictor_df))
    print(predictor_df.dtypes)

    continous_df, categorical_df = categorizing_data(predictor_df, response_df)
    # correlation_matrix(predictor_df)

    # Linear Regression

    # add constant to predictor variables
    X = sm.add_constant(continous_df)

    # create linear regression model
    model = sm.OLS(response_df, X)

    # fit the model
    results = model.fit()

    # print the coefficients of the model
    print("Coefficients: ", results.params)

    # print the t-values and p-values
    print("t-values: ", results.tvalues)
    print("p-values: ", results.pvalues)

    tvalues = results.tvalues
    pvalues = results.pvalues

    tvalues_trace = go.Bar(x=tvalues.index, y=tvalues.values, name="t-values")

    # create a trace for the p-values
    pvalues_trace = go.Bar(x=pvalues.index, y=pvalues.values, name="p-values")

    # create the layout for the plot
    layout = go.Layout(
        title="Regression Results",
        xaxis=dict(title="Coefficient"),
        yaxis=dict(title="Value"),
    )

    # create the figure and add the traces and layout
    fig = go.Figure(data=[tvalues_trace, pvalues_trace], layout=layout)

    # show the plot
    fig.show()

    # print the summary statistics
    print(results.summary())

    # regression_models(predictor_df, response_df, continous_df, categorical_df)
    res_pred_pair = [(i, df.loc[:, df.columns != i].columns.to_list()) for i in df]
    for i, j in res_pred_pair:
        continous_df, categorical_df = categorizing_data(df[j], df[i])
        # Finding the correlation heatmap agains each predictor variables
        correlation_matrix(df[j], df[i])

    # Mapping the datatypes and executing the corresponding functions to create a map
    predictor_response_map = {
        ("categorical", "boolean"): plot_categorical_predictor_with_boolean_response,
        ("continuous", "boolean"): plot_continuous_predictor_with_boolean_response,
        (
            "categorical",
            "continuous",
        ): plot_categorical_predictor_with_continuous_response,
        (
            "continuous",
            "continuous",
        ): plot_continuous_predictor_with_continuous_response,
    }

    # Iterating through the columns and mapping them to the correspnding functions to plot the graphs.
    for i in range(len(res_pred_pair[0][1])):
        for j in range(len(res_pred_pair[i][1])):
            feature = df[res_pred_pair[i][1][j]]
            if (df[res_pred_pair[i][1][j]].dtypes == "int64") and len(
                set(df[res_pred_pair[i][0]])
            ) == 2:
                response_type = "boolean"
                predictor_type = "categorical"
            elif (
                df[res_pred_pair[i][1][j]].dtypes != "int64"
                and len(set(df[res_pred_pair[i][0]])) == 2
            ):
                response_type = "boolean"
                predictor_type = "continuous"
            elif (
                df[res_pred_pair[i][1][j]].dtypes == "int64"
                and len(set(df[res_pred_pair[i][0]])) != 2
            ):
                response_type = "continuous"
                predictor_type = "categorical"
            else:
                predictor_type = "continuous"
                response_type = "continuous"
                function = predictor_response_map.get((predictor_type, response_type))
                function(
                    feature,
                    res_pred_pair[i][1][j],
                    df[res_pred_pair[i][0]],
                    res_pred_pair[i][0],
                )
                mean_square_error(
                    df.iloc[:, :-1], df.iloc[:, :-1].columns, df["home_team_wins"]
                )

    regression_models(predictor_df, response_df, continous_df, categorical_df)

    # Calculating the ranks of each columns and printing them in the list format
    n_features = predictor_df.shape[1]
    max_score = 0
    best_features = []
    for i in range(1, n_features + 1):
        for combo in itertools.combinations(predictor_df.columns, i):
            X = predictor_df[list(combo)]
            y = response_df
            model = LinearRegression().fit(X, y)
            score = model.score(X, y)
            if score > max_score:
                max_score = score
                best_features = list(combo)
            print("LIST OF BEST FEATURES -")
            print(best_features)


if __name__ == "__main__":
    main()

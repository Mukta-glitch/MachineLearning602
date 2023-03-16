import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import datasets
from statsmodels.formula.api import ols
from scipy.stats import f
import plotly.express as px
from sklearn.utils.multiclass import type_of_target as tot
import statsmodels

def col_is_bool(df, col):
    c = len(df[col].unique())
    if c == 2:
        True
    else:
        False
    return col
    
def plot_predictor(df, predictor_col):
    
    if df[predictor_col].dtype == object:
        # plot a histogram of the categorical variable
        df[predictor_col].value_counts().plot(kind='bar')
        print(f"{predictor_col} (Categorical)")
        px.violin()
    elif df[predictor_col].dtype in [int, float]:
        # plot a histogram of the continuous variable
        px.hist()
        print(f"{predictor_col} (Continuous)")
    else:
        print(f"Unknown data type for predictor column {predictor_col}")


def Con_Con(predictor_df, response_series, weights=None):
    results = {}
    response_mean = response_series.mean()
    for col in predictor_df.columns:
        predictor = predictor_df[col]

        # Calculate rankings using different algorithms
        rank_mean = predictor.rank(pct=True)
        rank_abs = predictor.abs().rank(pct=True)
        rank_minmax = (predictor - predictor.min()) / (predictor.max() - predictor.min())
        rank_zscore = predictor.sub(predictor.mean()).div(predictor.std())

        # Calculate differences with mean response
        diff_mean = rank_mean - response_mean
        diff_abs = rank_abs - response_mean
        diff_minmax = rank_minmax - response_mean
        diff_zscore = rank_zscore - response_mean

        # Add plots to results dictionary
        plot_mean = plot_heatmap(diff_mean, f"{col}_mean_diff")
        plot_abs = plot_heatmap(diff_abs, f"{col}_abs_diff")
        plot_minmax = plot_heatmap(diff_minmax, f"{col}_minmax_diff")
        plot_zscore = plot_heatmap(diff_zscore, f"{col}_zscore_diff")

        results[col] = {
            "diff_mean": diff_mean,
            "diff_abs": diff_abs,
            "diff_minmax": diff_minmax,
            "diff_zscore": diff_zscore,
            "plot_mean": plot_mean,
            "plot_abs": plot_abs,
            "plot_minmax": plot_minmax,
            "plot_zscore": plot_zscore,
        }
    return results
      

def Cat_cat(predictors, response, weights=None):
    results = {}
    for col in predictors.columns:
        predictor = predictors[col]
        if weights is not None:
            # Calculate weighted mean of response variable
            mean_response = np.average(response, weights=weights)
        else:
            mean_response = response.mean()

        # Calculate rankings using different algorithms
        rank_mean = predictor.rank(pct=True)
        rank_abs = predictor.abs().rank(pct=True)
        rank_minmax = (predictor - predictor.min()) / (predictor.max() - predictor.min())
        rank_zscore = predictor.sub(predictor.mean()).div(predictor.std())

        # Calculate differences with mean response
        diff_mean = rank_mean - mean_response
        diff_abs = rank_abs - mean_response
        diff_minmax = rank_minmax - mean_response
        diff_zscore = rank_zscore - mean_response

        # Add plots to results dictionary
        plot_mean = plot_heatmap(diff_mean, f"{col}_mean_diff")
        plot_abs = plot_heatmap(diff_abs, f"{col}_abs_diff")
        plot_minmax = plot_heatmap(diff_minmax, f"{col}_minmax_diff")
        plot_zscore = plot_heatmap(diff_zscore, f"{col}_zscore_diff")

        results[col] = {
            "diff_mean": diff_mean,
            "diff_abs": diff_abs,
            "diff_minmax": diff_minmax,
            "diff_zscore": diff_zscore,
            "plot_mean": plot_mean,
            "plot_abs": plot_abs,
            "plot_minmax": plot_minmax,
            "plot_zscore": plot_zscore,
        }
    return results
    
    
def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return


def plot_heatmap(data, title):
    fig = go.Figure(data=go.Heatmap(z=data, zmin=data.min(), zmax=data.max()))
    fig.update_layout(
        title=title,
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    file_name = f"Heatmap.html"
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )
    return file_name

def linear_regression(y, predictor, column):
    linear_regression_model = statsmodels.api.OLS(y, predictor)
    linear_regression_model_fitted = linear_regression_model.fit()
    print(f"Feature_Name: {column}")
    print(linear_regression_model_fitted.summary())

    # Get statistics
    tval = round(linear_regression_model_fitted.tvalues[1], 6)
    pval = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    return tval, pval
    
def logistic_regression(y, predictor, feature):
    logistic_regression_model = statsmodels.api.Logit(y, predictor)
    logistic_regression_model_fitted = logistic_regression_model.fit()
    print(f"Feature_Name:{feature}")
    print(logistic_regression_model_fitted.summary())

    # Get the statistics
    tval = round(logistic_regression_model_fitted.tvalues[1], 6)
    pval = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])

    return tval, pval
    
def anova_pvalue(df, predictor_col, response_col):
    """
    Calculate ANOVA p-value for a given predictor column and response column
    """
    model = ols(f"{response_col} ~ {predictor_col}", data=df).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    p_value = aov_table["PR(>F)"][0]
    return p_value


def cat_response_cat_predictor(df, predictor_col, response_col):

    #Create heatmap for categorical predictor and response variables
    # Check if the response variable is binary
    if col_is_bool(df, response_col):
        results = Cat_cat(df[predictor_col], df[response_col])
        for col, data in results.items():
            file_name = data["plot_mean"]
            display(HTML(f"<h3>{col}</h3>"))
            display(HTML(f'<iframe src="{file_name}" width="100%" height="500px"></iframe>'))
    else:
        # Calculate ANOVA p-values for each predictor column
        p_values = []
        for col in predictor_col:
            p_value = anova_pvalue(df, col, response_col)
            p_values.append(p_value)

        # Create dataframe of p-values
        p_values_df = pd.DataFrame(
            {
                "predictor": predictor_col,
                "p_value": p_values,
            }
        )

        # Create bar chart of p-values
        fig = go.Figure(
            [go.Bar(x=p_values_df["predictor"], y=p_values_df["p_value"], marker_color="salmon")]
        )
        fig.update_layout(
            title="ANOVA p-values for Categorical Predictor and Response",
            xaxis_title="Predictor",
            yaxis_title="p-value",
        )
        file_name = f"ANOVA_categorical_{response_col}.html"
        fig.write_html(
            file=file_name,
            include_plotlyjs="cdn",
        )
        display(HTML(f'<iframe src="{file_name}" width="100%" height="500px"></iframe>'))
        
def main():
    dataset1 = datasets.load_iris()
    dataset2 = datasets.load_digits()
    dataset3 = datasets.load_diabetes()
    dataset7 = datasets.load_wine()

    print("Choose the dataset you want :1.IRIS 2.DIGITS 3.DIABETES 4.BREAST CANCER")
    i = input("Enter >")
    i = i.lower()
    if i == "iris":
        dataset = dataset1
    elif i == "digits":
        dataset = dataset2
    elif i == "diabetes":
        dataset = dataset3
    elif i == "breast cancer":
        dataset = dataset5
    else:
        print("Invalid input.")
        return
    
    df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    df['target'] = pd.Series(dataset.target)
    df.head(5)
    
    col_list = df.columns.values.tolist()
    print_heading("Dataset")
    print(df)
    
    print_heading("Response")
    response=dataset.target
    print(response)

    # Perform ANOVA
    '''
    print_heading("ANOVA Results")
    anova_results = anova(df, 'target')
    print(anova_results)
	
    # Plot predictor variables
    print_heading("Predictor Plots")
    for col in col_list[:-1]:
        plot_predictor(df, col_list)
        
    '''

    # Perform linear regression
    print_heading("Linear Regression Results")
    predictors = df.iloc[:, :-1]
    response = df.iloc[:, -1]
    linear_regression(response, predictors, 'target')
    
    print_heading("Logistic Regression Results")
    predictors = df.iloc[:, :-1]
    response = df.iloc[:, -1]
    logistic_regression(response, predictors, 'target')
    

    # Perform categorical to categorical comparison
    print_heading("Categorical-Categorical Comparison")
    bool_cols = [col for col in df.columns if col_is_bool(df, col)]
    if len(bool_cols) >= 2:
        predictors = df[bool_cols]
        response = df.iloc[:, -1]
        cat_cat_results = Cat_cat(predictors, response)
        print(cat_cat_results)
    else:
        print("Not enough boolean columns for categorical-categorical comparison.")
        
     #Perform Continous to continous comparison
    print_heading("Contonous-Continous Comparison")
    con_cols = [col for col in df.columns]
    if len(con_cols) >2:
        predictors = df[con_cols]
        response = df.iloc[:, -1]
        Con_Con_results = Con_Con(predictors, response)
        print(Con_Con_results)
    else:
        print("Invalid")
        
if __name__ == "__main__":
    sys.exit(main())

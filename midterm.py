"""
REFERENCES-
1. https://plotly.com/python-api-reference/generated/plotly.express.imshow.html
2. https://plotly.com/python/heatmaps/
3. https://www.statsmodels.org/stable/index.html
4. https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html
5. https://www.statsmodels.org/dev/anova.html

"""


import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from scipy import stats
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import type_of_target as tot
import statsmodels.api as sm
import ipywidgets as widgets
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns

def col_is_bool(df,col):
    c = len(df[col].unique())
    if c == 2:
        True
    else:
        False
    return col
    
def plot_predictor(df,col):
    if df[col]==object or df[col]==str:
        True
    else:
        False   
    return col

def anova_pvalue(df, predictor_col, response_col):
    """
    Calculate ANOVA p-value for a given predictor column and response column
    """
    model = ols(f"{response_col} ~ {predictor_col}", data=df).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    p_value = aov_table["PR(>F)"][0]
    return p_value

# PLOT Functions
def adj_plot_hist (pred, initial_bin_width=3): 
    pop_mean = [pred.mean()] * len(pred)
    figure_widget = go.FigureWidget(
        data=[go.Histogram(x=pred, xbins={"size": initial_bin_width},
                           name='Bin'),
              go.Scatter(x=pred[0:], y=pop_mean,
                         mode='lines',
                         name='Predictor Mean')]
    )

    bin_slider = widgets.FloatSlider(
        value=initial_bin_width,
        min=0,
        max=pred.max(),
        step=pred.max() / 30,
        description="Bin width:",
        readout_format=".1f", 
    )

    histogram_object = figure_widget.data[0]

    def set_bin_size(change):
        histogram_object.xbins = {"size": change["new"]}

    bin_slider.observe(set_bin_size, names="value")

    output_widget = widgets.VBox([figure_widget, bin_slider])
    return output_widget

def plot_heatmap(to_plot):
    user_db=to_plot
    fig = go.Figure(data=go.Heatmap(z=user_db, zmin=user_db.min(), zmax=user_db.max()))
    fig.update_layout(
        title=title,
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    file_name = f"Heatmap.html"
    fig.show()

def plot_hist (pred): 
    pop_mean = [pred.mean()] * len(pred)
    data=[go.Histogram(x=pred, xbins={"size": len(pred.unique())/20},
                       name='Bin'),
          go.Scatter(x=pred[0:], y=pop_mean,
                     mode='lines',
                     name='Predictor Mean')]
    return

def plot_cont_cont(to_plot):
    user_db = to_plot
    X = user_db.data
    y = user_db.target
    con_con_file_list = []
    tp_list = []
    for idx, column in enumerate(X.T):
        feature_name = user_db.feature_names[idx]
        feature_name = feature_name.replace("/","-")
        predictor = sm.add_constant(column)
        linear_regression_model = sm.OLS(y, predictor)
        linear_regression_model_fitted = linear_regression_model.fit()
        print(f"Variable: {feature_name}")
        print(linear_regression_model_fitted.summary())
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

        fig = px.scatter(x=column, y=y, trendline="ols")
        fig.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
        file_name = f"HW4_{feature_name}_lin_plot.html"
        fig.write_html(file=file_name, include_plotlyjs="cdn")
        con_con_file_list += [file_name]
        tp_list += [f'{p_value} & {t_value}']

    return con_con_file_list, tp_list


def plot_cont_cat(to_plot):
    # Violin plot on predictor grouped by response
    user_db = to_plot
    X = user_db.data
    y = user_db.target
    con_cat_file_list=[]
    #
    for idx, column in enumerate(X.T):
        feature_name = user_db.feature_names[idx]
        feature_name = feature_name.replace("/","-")
        predictor = sm.add_constant(column)
        logistic_regression_model = LogisticRegression(random_state=4622).fit(
            predictor, y
        )
        print(f"Variable: {feature_name}")
        print(logistic_regression_model.score(predictor, y))
        the_score = logistic_regression_model.score(predictor, y)
        fig = px.area(x=column, y=y,color=column)
        fig.update_layout(
            title=f"Variable:{feature_name},Logistic Regression Score={the_score}",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
        file_name = f"HW4_{feature_name}_cont_cat_area_plot.html"
        fig.write_html(file=file_name, include_plotlyjs="cdn")
        con_cat_file_list += [file_name]
    return con_cat_file_list 


def plot_cat_cat(to_plot):

    user_db = to_plot
    X = user_db.data
    y = user_db.target
    cat_cat_file_list=[]
    for idx, column in enumerate(X.T):
        feature_name = user_db.feature_names[idx]
        feature_name = feature_name.replace("/","-")
        predictor = sm.add_constant(column)
        logistic_regression_model = LogisticRegression(random_state=4622).fit(
            predictor, y
        )
        print(f"Variable: {feature_name}")
        print(logistic_regression_model.score(predictor, y))
        the_score = logistic_regression_model.score(predictor, y)
        fig = px.violin(x=y, y=column)
        fig.update_layout(
            title=f"Variable:{feature_name},Logistic Regression Fit Score={the_score}",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
        file_name = f"HW4_{feature_name}_cat_cat_violin_plot.html"
        fig.write_html(file=file_name, include_plotlyjs="cdn")
        cat_cat_file_list += [file_name]
    return cat_cat_file_list
    
def plot_cat_cat(to_plot):

    user_db = to_plot
    X = user_db.data
    y = user_db.target
    cat_cat_file_list=[]
    for idx, column in enumerate(X.T):
        feature_name = user_db.feature_names[idx]
        feature_name = feature_name.replace("/","-")
        predictor = sm.add_constant(column)
        logistic_regression_model = LogisticRegression(random_state=4622).fit(
            predictor, y
        )
        print(f"Variable: {feature_name}")
        print(logistic_regression_model.score(predictor, y))
        the_score = logistic_regression_model.score(predictor, y)
        fig = px.violin(x=y, y=column)
        fig.update_layout(
            title=f"Variable:{feature_name},Logistic Regression Fit Score={the_score}",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
        file_name = f"HW4_{feature_name}_cat_cat_violin_plot.html"
        fig.write_html(file=file_name, include_plotlyjs="cdn")
        cat_cat_file_list += [file_name]
    return cat_cat_file_list

def plot_cat_cont(to_plot):

    user_db = to_plot
    X = user_db.data
    y = user_db.target
    cat_con_file_list=[]
    #
    for idx, column in enumerate(X.T):
        feature_name = user_db.feature_names[idx]
        feature_name = feature_name.replace("/","-")
        predictor = sm.add_constant(column)
        logistic_regression_model = LogisticRegression(random_state=4622).fit(
            predictor, y
        )
        print(f"Variable: {feature_name}")
        print(logistic_regression_model.score(predictor, y))
        the_score = logistic_regression_model.score(predictor, y)
        
        #df = pd.DataFrame(np.random.random((2,442)), columns=y)
        #fig = sns.heatmap(df)
        #fig.imshow(df)
        fig1 = px.scatter(x=column, y=y, color=column)
        fig1.update_layout(
            title=f"Variable:{feature_name},Logistic Regression Fit Score={the_score}",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
        file_name = f"HW4_{feature_name}__cat_cont_Scatter.html"
        fig1.write_html(file=file_name, include_plotlyjs="cdn")
        cat_con_file_list += [file_name]
    return cat_con_file_list

def plot_bool(to_plot):
    user_db = to_plot
    X = user_db.data
    y = user_db.target
    bool_file_list = []
    for idx, column in enumerate(X.T):
        feature_name = user_db.feature_names[idx]
        predictor = sm.add_constant(column)
        log_regression_model = sm.Logit(y, predictor)
        log_regression_model_fitted = log_regression_model.fit()
        print(f"Variable: {feature_name}")
        print(log_regression_model_fitted.summary())

        t_value = round(log_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(log_regression_model_fitted.pvalues[1])

        fig = px.scatter(x=column, y=y, trendline="ols")
        fig.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
        file_name = f"HW4_{feature_name}_bool_plot.html"
        fig.write_html(file=f"HW4_{feature_name}log_plot.html", include_plotlyjs="cdn")
        bool_file_list += [file_name]
    return bool_file_list


def predictor_select(ds):
    predictors = ds.feature_names
    selection_df = pd.DataFrame(predictors, columns=["Predictor"])
    selection_df.index += 1
    print(selection_df)
    pred_choice = input(
        "Which predictors will you use? (use index numbers separate by a comma) :"
    )
    pred_choice = pred_choice.split(",")
    for i in range(0, len(pred_choice)):
        pred_choice[i] = int(pred_choice[i])
    feature_list = [selection_df.at[i, "Predictor"] for i in pred_choice]

    return feature_list


def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return

def list_to_df(from_df, to_df, col):
    temp_list = from_df[col].to_list()
    to_df[col] = pd.Series(temp_list, index=to_df.index)

def clickable_report(loc):
    name= os.path.basename(loc)
    return '<a target="_blank" href=\"{}\">{}</a>'.format(loc,name)

def coor_con_con(df, col, resp):
    corr_con_con_list =[]
    a = df[col]
    b = df[resp]
    corr_con_con_list += [stats.pearsonr(a,b)]
    return corr_con_con_list

def coor_con_cat(df, col, resp):
    corr_con_cat_list =[]
    a = df[col]
    b = df[resp]
    corr_con_cat_list += [stats.kendalltau(a,b)]
    return corr_con_cat_list

# Main functions
def main():
    start_t = datetime.now()
    ds1 = datasets.load_iris()
    ds2 = datasets.load_diabetes()
    ds3 = datasets.load_digits()
    ds4 = datasets.load_wine()
    ds5 = datasets.load_breast_cancer()

    print(
        """ Chose Dataset-1 - IRIS 2 - DIABETES 3 - DIGITS 4 - WINE 5 - BREAST CANCER  6 - user choice-secret dataset
                  """
    )

    choice = int(input("Which set? ")or 5)

    if choice == 1:
        the_ds = ds1
    elif choice == 2:
        the_ds = ds2
    elif choice == 3:
        the_ds = ds3
    elif choice == 4:
        the_ds = ds4
    elif choice == 5:
        the_ds = ds5
    elif choice == 6:
        the_ds = input("Input the path to dataset! ")
    elif choice == "":
        the_ds = ds4
    else:
        print(f"{choice} YOUR CHOICE IS INVALID ")

    print(f"You chose {choice}")
    working_df = pd.DataFrame(the_ds.data, columns=the_ds.feature_names)
    working_df["target"] = pd.Series(the_ds.target)  
    working_df.head()

    col_list = working_df.columns.values.tolist()
    print_heading("Original Dataset")
    print(working_df)

    resp = input('Select the desired response -|default = target ') or 'target'

    res_type = tot(working_df[resp])
    print_heading("Response type is " + res_type)

    type_mask = [tot(working_df[i]) for i in col_list]
    bool_list = [col_is_bool(working_df,i) for i in col_list]
    bool_df = pd.DataFrame(bool_list,index=[col_list])
    predictor_array = np.column_stack((col_list, type_mask))
    pred_df = pd.DataFrame(predictor_array, columns=["Predictor", "Category"])


    cont_feature_df = pred_df[
        pred_df["Category"] == "continuous"
        ]  
    try:
        cont_feature_df = cont_feature_df.drop(
            "target", axis=1, inplace=True
        )
    except Exception:
        pass

    cat_feature_df = pred_df[
        pred_df["Category"] != "continuous"
        ]  
    try:
        cat_feature_df = cat_feature_df.drop(
            "target", axis=1, inplace=True
        )
    except Exception:
        pass
        
      

    binary_feature_df = pred_df[
        pred_df["Category"] == "binary"
        ]  
    try:
        binary_feature_df = binary_feature_df.drop(
            "target", axis=1, inplace=True
        )
    except Exception:
        pass

    
    bool_feature_df = bool_df.dropna()
    try:
        bool_feature_df = bool_feature_df.drop(
            "target", axis=1, inplace=True
        )
    except Exception:
        pass

    print("No Continuous!") if cont_feature_df.empty else print(cont_feature_df)
    print("No Categorical!") if cat_feature_df.empty else print(cat_feature_df)
    print("No Boolean!") if binary_feature_df.empty else print(binary_feature_df)

    cont_feature_list = list(cont_feature_df["Predictor"])
    cat_feature_list = list(cat_feature_df["Predictor"])

    report_col = (
        "CATEGORY",
        "p-val_&_t-val",
        "REGRESSION",
        "LOGISTIC REGRESSION",
        "CATEGORICAL VS CATEGORICAL",
        "CONTINOUS VS CATEGORICAL",
    )
    report_df = pd.DataFrame("", index=col_list, columns=report_col)
    report_df = report_df.drop(['target'])
    # 2
    report_col_2 = ('CORRELATION CONTINOUS VS CONTINOUS',
                    'CORRELATION CONTINOUS VS CATEGORICAL',)
    report_df_2 = pd.DataFrame("", index=col_list, columns=report_col_2)
    pred_df = pred_df.set_index(["Predictor"])
    pred_df = pred_df.drop(['target'])


    report_df.index.name = "Predictor"
    pred_df.index = report_df.index

    report_style = report_df.style

    con_cat_file_list = plot_cont_cat(the_ds)
    temp_df = pd.DataFrame(con_cat_file_list, columns=['CONTINOUS VS CATEGORICAL'])
    list_to_df(temp_df, report_df, "CONTINOUS VS CATEGORICAL")
    


    con_con_file_list, tp_list = plot_cont_cont(the_ds)
    temp_df = pd.DataFrame(con_con_file_list, columns=['REGRESSION'])
    list_to_df(temp_df, report_df, "REGRESSION")

    temp_df = pd.DataFrame(tp_list, columns=['p-val_&_t-val'])
    list_to_df(temp_df, report_df, "p-val_&_t-val")

    cat_con_file_list = plot_cat_cont(the_ds)
    temp_df = pd.DataFrame(cat_con_file_list, columns=['LOGISTIC REGRESSION'])
    list_to_df(temp_df, report_df, "LOGISTIC REGRESSION")

    cat_cat_file_list = plot_cat_cat(the_ds)
    temp_df = pd.DataFrame(cat_cat_file_list, columns=['CATEGORICAL VS CATEGORICAL'])

    list_to_df(temp_df, report_df, "CATEGORICAL VS CATEGORICAL")

    corr_con_con_list = [coor_con_con(working_df, i, 'target') for i in col_list]
    list_to_df(pd.DataFrame({'CORRELATION CONTINOUS VS CONTINOUS': corr_con_con_list}), report_df_2, 'CORRELATION CONTINOUS VS CONTINOUS')

    corr_con_cat_list = [coor_con_cat(working_df, i, 'target') for i in col_list]
    list_to_df(pd.DataFrame({'CORRELATION CONTINOUS VS CATEGORICAL': corr_con_cat_list}), report_df_2, 'CORRELATION CONTINOUS VS CATEGORICAL')

    corr_con_con_list = [coor_con_con(working_df, i, 'target') for i in col_list]
    temp_df = pd.DataFrame(corr_con_con_list, columns=['CORRELATION CONTINOUS VS CONTINOUS'])
    list_to_df(temp_df, report_df_2, 'CORRELATION CONTINOUS VS CONTINOUS')

    coor_con_cat_list = [coor_con_cat(working_df, i, 'target') for i in col_list]
    temp_df = pd.DataFrame(corr_con_con_list, columns=['CORRELATION CONTINOUS VS CATEGORICAL'])
    list_to_df(temp_df, report_df_2, 'CORRELATION CONTINOUS VS CATEGORICAL')

    report_df_2 = report_df_2.sort_values(['CORRELATION CONTINOUS VS CONTINOUS'], ascending=[False])
    report_df_2 = report_df_2.drop(['target'])
    report_df_2.to_html("REPORT 2" + datetime.now().strftime("%Y_%m_%d-%H_%M") + ".html")

    list_to_df(pred_df, report_df, "Category")

    cols_to_format = ['REGRESSION']
    html = report_df.to_html(escape=True, render_links=True)
    report_style = report_df.style
    for col in cols_to_format:
        report_style = report_style.format({col: clickable_report})
    c=['LOGISTIC REGRESSION']
    for col in c:
        report_style = report_style.format({col: clickable_report})
    d=['CATEGORICAL VS CATEGORICAL']
    for col in d:
        report_style = report_style.format({col: clickable_report})
    e=['CONTINOUS VS CATEGORICAL']
    for col in e:
        report_style = report_style.format({col: clickable_report})

    report = report_style.render()
   
    with open("REPORT 1" + datetime.now().strftime("%Y_%m_%d-%H_%M") + ".html", "w") as temp:
        temp.write(report)

    print(datetime.now() - start_t)

if __name__ == "__main__":
    sys.exit(main())
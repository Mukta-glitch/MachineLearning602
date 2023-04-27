import sys
from pyspark import StorageLevel, keyword_only
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, lit, split, when
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import round
import pyspark.sql.functions as F
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os
import itertools
import warnings

def categorizing_data(predictor_df, response_df):
	if len(set(response_df))== 2:
	    response_type = 'Boolean'
	    print(response_type)
	else:
	    response_type = 'Continuous'
	    print(response_type)

	# loop through each predictor column
	for predictor_col in predictor_df.columns:
	    # determine if predictor is cat/cont
	    if predictor_df[predictor_col].dtypes == 'object':
	        predictor_type = 'Categorical'
	        print(predictor_col+ " TYPE OF PREDICTOR IS - "+predictor_type)
	        fig = px.histogram(predictor_df, color=predictor_col, x=response_df['home_team_wins'], barmode='group')
	        fig.update_layout(title=f'{predictor_col} vs home_team_wins', xaxis_title=predictor_col, yaxis_title='home_team_wins')
	        fig.show()
	    else:
	        predictor_type = 'Continuous'
	        print(predictor_col+ " TYPE OF PREDICTOR IS - "+predictor_type)
	        fig = px.scatter(predictor_df,x=predictor_col,y=response_df['home_team_wins'],color=predictor_col,trendline='ols')
	        fig.update_layout(title=f'{predictor_col} vs home_team_wins', xaxis_title=predictor_col, yaxis_title='home_team_wins')
	        fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGray'),
	                              opacity=0.8,
	                              symbol='circle',
	                              line_width=1,
	                              sizemode='diameter'))
	        fig.show()

 	# Split a dataframe of predictors into categorical and continuous dataframes
	categorical_arr = predictor_df.select_dtypes(include="object").columns.tolist()
	continous_arr = predictor_df.select_dtypes(exclude="object").columns.tolist()
	categorical_df = predictor_df[categorical_arr]
	continous_df = predictor_df[continous_arr]
	print("categorical predictors are:\n", categorical_arr)
	print("continuous predictors are:\n", continous_arr)
	return(continous_df, categorical_df)

# Finding the correlation heatmap agains each predictor variables
def correlation_matrix(predictor_df):
	corr_matrix = predictor_df.corr()
	print(corr_matrix)
	fig = px.imshow(corr_matrix,
	                x=corr_matrix.columns,
	                y=corr_matrix.columns,
	                color_continuous_scale='RdBu')
	fig.update_layout(xaxis_title="Predictor Variables",
	                  yaxis_title="Predictor Variables",title="Correlation Matrix of Predictor values")
	fig.show()


# Define the models and their corresponding hyperparameters
def regression_models(predictor_df, response_df, continous_df, categorical_df):
	models = {
	'Linear Regression': {
	    'model': LinearRegression(),
	    'param_grid': {}
	},
	'Support Vector Machine': {
	    'model': SVR(),
	    'param_grid': {
	        'model__C': [0.1, 1, 10],
	        'model__kernel': ['linear', 'rbf', 'poly'],
	        'model__degree': [2, 3, 4],
	    }
	},
	'Decision Tree': {
	    'model': DecisionTreeRegressor(),
	    'param_grid': {
	        'model__max_depth': [10, 20, 30],
	        'model__min_samples_leaf': [1, 2, 4],
	    }
	},
	'Random Forest': {
	    'model': RandomForestRegressor(),
	    'param_grid': {
	        'model__n_estimators': [50, 100, 150],
	        'model__max_depth': [10, 20, 30],
	        'model__min_samples_leaf': [1, 2, 4],
	    }
	}
	}

	print(response_df.columns, predictor_df.columns)
	X = predictor_df
	y = response_df['home_team_wins']

	# split predictor dataframe into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42)

	# preprocess the data
	continuous_transformer = Pipeline([
	('scaler', StandardScaler())
	])
	categorical_transformer = Pipeline([
	('encoder', OneHotEncoder(handle_unknown='ignore'))
	])
	preprocessor = ColumnTransformer(transformers=[
	('continuous', continuous_transformer, continous_df.columns),
	('categorical', categorical_transformer, categorical_df.columns)
	])

	# iterate over the models and their corresponding hyperparameters
	for name, model_params in models.items():
		print(f'Training {name}')
		model = model_params['model']
		param_grid = model_params['param_grid']

		# create a pipeline that includes preprocessing and the current model
		pipeline = Pipeline([
		    ('preprocessor', preprocessor),
		    ('model', model)
		])

		# tune hyperparameters of the pipeline using cross-validation
		cv = GridSearchCV(pipeline, param_grid, cv=3)
		cv.fit(X_train, y_train)
		print(f'Best parameters for {name}:', cv.best_params_)
		print(f'Best score for {name}:', cv.best_score_)

		# train the pipeline on the entire dataset using the best hyperparameters
		pipeline.set_params(**cv.best_params_)
		pipeline.fit(predictor_df, response_df)

		# evaluate the pipeline on the testing set
		y_pred = pipeline.predict(X_test)
		print(f'Mean squared error for {name}:', mean_squared_error(y_test, y_pred))
		print('R2 score:', r2_score(y_test, y_pred))

#SVM gives better results as mean squared error is lowest - Mean squared error for Support Vector Machine: 0.04871777453323358

# Plotting all the maps based on data types of the columns
def plot_categorical_predictor_with_boolean_response(df, feature_names, y, response):
    df_plot = df[[response, feature_names]]
    fig = px.density_heatmap(df_plot, x=feature_names, y=response)
    fig.update_xaxes(title=column)
    fig.update_yaxes(title=response)

    file_name = f"hw_5_plot/{feature_names}.html"
    fig.write_html(file_name, include_plotlyjs="cdn")

    file_name_ = f"hw_5_plot/continuous_{feature_names}.html"
    fig.write_html(file_name_, include_plotlyjs="cdn")

def plot_continuous_predictor_with_boolean_response(df, feature_names, y, response):

    fig = px.histogram(
        df, 
        x=feature_names, 
        color=response, 
        nbins=20, 
        barmode='overlay', 
        opacity=0.7,
        histnorm='percent'
    )
    fig.update_layout(
        title=f"{column.capitalize()} Distribution by Boolean Response",
        xaxis_title=column.capitalize(),
        yaxis_title="Density"
    )
    file_name = f"hw_5_plot/continuous_{feature_names}.html"
    fig.write_html(file_name, include_plotlyjs="cdn")

def plot_categorical_predictor_with_continuous_response(feature, feature_names, y, response):
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
    file_name = f"plot/hw_5_plot/categorical_{feature_names}.html"
    file_name_ = f"hw_05_plot{feature_names}.html"
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )
    fig.write_html(
        file=file_name_,
        include_plotlyjs="cdn",
    )


# Function for plotting scatter plot
def plot_continuous_predictor_with_continuous_response(feature, feature_names, y, response):
    if isinstance(feature, pd.DataFrame):
        print("feature is a DataFrame object")
        feature = feature[feature_names]
    else:
    	df = pd.DataFrame()
    	df[feature_names] = feature
    	df[response] = y
    	fig = px.scatter(df, x=feature_names, y=response,trendline='ols')
    	fig.update_layout(title="Continuous Response by Continuous Predictor")
    	fig.show()
    	file_name = f"plot/hw_5_plot/cr_con_{feature_names}.html"
    	file_name_ = f"plot/hw_5_plot/categorical_con_{feature_names}.html"
    	fig.write_html(file=file_name, include_plotlyjs="cdn",)
    	fig.write_html(file=file_name_, include_plotlyjs="cdn",)

# Logistic regression
def logistic_regression(y, pred, column):
    model = statsmodels.api.Logit(y, pred)
    fitted_model = model.fit()
    print(f"Feature Name: {feature_names}")
    print(fitted_model.summary())
    return round(fitted_model.tvalues[1], 6), fitted_model.pvalues[1], f"Variable: {feature_names}"

def calculate_ranking(data, weights=None):
    # calculate the mean response for each combination of factors
    if weights is None:
        bin_means = data.groupby(["factor1", "factor2"])["response"].mean().reset_index()
    else:
        data["weighted_response"] = data["response"] * weights
        bin_means = data.groupby(["factor1", "factor2"])["weighted_response"].sum() / data.groupby(["factor1", "factor2"])[weights].sum()
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
    norm = mpl.colors.Normalize(vmin=bin_means["response"].min(), vmax=bin_means["response"].max())
    sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, shrink=0.5, aspect=5, label="Response")
    ax.set_zlim(pop_mean - 2 * (pop_mean - bin_means["response"].min()),
                pop_mean + 2 * (bin_means["response"].max() - pop_mean))

    # add labels and title to the plot
    ax.set_xlabel("Factor 1")
    ax.set_ylabel("Factor 2")
    ax.set_zlabel("Response")
    ax.set_title("Bin Means as 3D Surface Plot")

    plt.show()


# x = predictor_df.index.values
# y = predictor_df.columns.values
# z = predictor_df.values
# mean_z = z.mean() # Population mean

# # Create the 3D surface plot
# fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])

# # Set the axis labels
# fig.update_layout(scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis'))

# # Scale the legend so that the center is at the population mean
# fig.update_layout(coloraxis_colorbar=dict(center=mean_z, lenmode='fraction', len=0.75))

# # Show the plot
# fig.show()


# residuals = predictor_df - predictor_df.mean().mean()

# # Define the data
# x = predictor_df.index.values
# y = predictor_df.columns.values
# z = residuals.values

# # Create the 3D surface plot
# fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])

# # Set the axis labels
# fig.update_layout(scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Residuals'))

# # Show the plot
# fig.show()


def main():
	if not os.path.exists("plot/hw_5_plot"):
		os.makedirs("plot/hw_5_plot")


	warnings.simplefilter(action='ignore')

	appName = "Pyspark Homework 3"
	master = "local"
	# Create Spark session
	spark = SparkSession.builder \
		.appName(appName) \
		.master(master) \
		.getOrCreate()
		
	# Define table from database and set JDBC connection 
	sql = "select rolling_Slugging_Percentage,rolling_Times_On_Base,rolling_Earned_Run_Average,rolling_On_Base_Percentage,rolling_HomeRuns_per_9_Innings,rolling_Walk_plus_Hits_per_Inning_Pitched,rolling_Strike_To_Walk_Ratio,rolling_away_wp,rolling_run_differential,rolling_home_wp,rolling_Batting_Average,rolling_innings_pitched from baseball.joined1"
	sql1="select home_team_wins from baseball.joined1"
	database = "baseball"
	user = "admin_user"
	password = "secret_password"
	server = "localhost"
	port = "3306"
	jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
	jdbc_driver = "org.mariadb.jdbc.Driver"

	# Create a data frame of batting_df by reading data from Mariadb via JDBC
	predictor = spark.read.format("jdbc") \
		.option("url", jdbc_url) \
		.option("query", sql) \
		.option("user", user) \
		.option("password", password) \
		.option("driver", jdbc_driver) \
		.load()


	predictor.show(10)

	response = spark.read.format("jdbc") \
		.option("url", jdbc_url) \
		.option("query", sql1) \
		.option("user", user) \
		.option("password", password) \
		.option("driver", jdbc_driver) \
		.load()

	response.show(10)
	#inning_df.persist(StorageLevel.MEMORY_AND_DISK)

	#Create a data frame of game by reading data from Mariadb via JDBC

	response_df = response.toPandas()
	predictor_df= predictor.toPandas()

	#joined_df = response_df.join(predictor_df, "team_id")
	print(response_df.head(5))
	print(predictor_df.head(5))
	#print(joined_df.head(5))
	print(len(response_df))
	print(len(predictor_df))
	print(predictor_df.dtypes)

	continous_df, categorical_df = categorizing_data(predictor_df, response_df)
	correlation_matrix(predictor_df)

	#Linear Regression

	# add constant to predictor variables
	X = sm.add_constant(continous_df)

	# create linear regression model
	model = sm.OLS(response_df, X)

	# fit the model
	results = model.fit()

	# print the coefficients of the model
	print('Coefficients: ', results.params)

	# print the t-values and p-values
	print('t-values: ', results.tvalues)
	print('p-values: ', results.pvalues)

	# print the summary statistics
	print(results.summary())

	regression_models(predictor_df, response_df, continous_df, categorical_df)

	# Mapping the datatypes and executing the corresponding functions to create a map
	predictor_response_map = {
	    ("categorical", "boolean"): plot_categorical_predictor_with_boolean_response,
	    ("continuous", "boolean"): plot_continuous_predictor_with_boolean_response,
	    ("categorical", "continuous"): plot_categorical_predictor_with_continuous_response,
	    ("continuous", "continuous"): plot_continuous_predictor_with_continuous_response
	}

	# Iterating through the columns and mapping them to the correspnding functions to plot the graphs.
	for index, column in enumerate(predictor_df):
	    feature = predictor_df[column]
	    print(feature)
	    print("Column: "+column)
	    if predictor_df[column].dtypes == 'object' and len(set(response_df))== 2:
	    	response_type = 'boolean'
	    	predictor_type = 'categorical'
	    else:
	    	predictor_type = 'continuous'
	    	response_type = 'continuous'
	    function = predictor_response_map.get((predictor_type, response_type))
	    function(feature, column,response_df['home_team_wins'], response_type)

	# Calculating the ranks of each columns and printing them in the list format
	n_features = predictor_df.shape[1]
	max_score = 0
	best_features = []
	for i in range(1, n_features+1):
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

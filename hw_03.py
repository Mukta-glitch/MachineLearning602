'''references=
1.https://www.crowdstrike.com/blog/deep-dive-into-custom-spark-transformers-for-machine-learning-pipelines/
2.https://spark.apache.org/docs/1.6.0/ml-guide.html
3.https://teaching.mrsharky.com/sdsu_fall_2020_lecture06.html#/7/11/5
4.https://www.youtube.com/watch?v=jEyahxFp3ak
5.https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa
6.https://api-docs.databricks.com/python/pyspark/latest/api/pyspark.ml.evaluation.BinaryClassificationEvaluator.html
'''


import sys
from pyspark import StorageLevel, keyword_only
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, lit, split, when
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import round
from pyspark.ml.linalg import Vectors

import warnings
warnings.simplefilter(action='ignore')

appName = "Pyspark Homework 3"
master = "local"
# Create Spark session
spark = SparkSession.builder \
	.appName(appName) \
	.master(master) \
	.getOrCreate()
# Define table from database and set JDBC connection 
sql = "select * from baseball.batter_counts"
sql1="select * from baseball.game"
database = "baseball"
user = "admin_user"
password = "secret_password"
server = "localhost"
port = "3306"
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"

# Create a data frame of batting_df by reading data from Mariadb via JDBC
batting_df = spark.read.format("jdbc") \
	.option("url", jdbc_url) \
	.option("query", sql) \
	.option("user", user) \
	.option("password", password) \
	.option("driver", jdbc_driver) \
	.load()

batting_df.select("batter", "game_id", "atBat", "Hit").show()
batting_df.createOrReplaceTempView("batter_view")
batting_df.persist(StorageLevel.MEMORY_AND_DISK)

#Create a data frame of game by reading data from Mariadb via JDBC

game_df = spark.read.format("jdbc") \
	.option("url", jdbc_url) \
	.option("query", sql1) \
	.option("user", user) \
	.option("password", password) \
	.option("driver", jdbc_driver) \
	.load()
	
game_df.select("game_id", "local_date").show()
game_df.createOrReplaceTempView("game")
game_df.persist(StorageLevel.MEMORY_AND_DISK)

#Initiate batter_view and calculate Hit/atBat value

bat_avg = spark.sql(
	"""
				SELECT
				batter
				, game_id
				, atBat
				, Hit
				, (Hit/nullif(atBat,0)) AS game_bat_avg
				, CASE WHEN Hit > 0 THEN 1 else 0 END AS AH
				FROM batter_view
				ORDER BY batter, game_id
	"""
	)
bat_avg.show()

# Rolling dataframe to calculate rolling batting average

rolling_df = spark.sql(
		"""
		        SELECT
	            b.batter
	            , b.game_id
	            , g.local_date
	            , SUM(b.Hit) AS Total_Hit
	            , SUM(b.atBat) AS Total_atBat
	            FROM game g JOIN
	            batter_view b ON
	            b.game_id = g.game_id AND
	            g.local_date BETWEEN DATE_SUB(g.local_date, 100) AND
	            g.local_date
	            GROUP BY
	            b.batter, b.game_id, g.local_date
		"""
	)
rolling_df.show()


#Transform() converts 1 dataframe to another , Here we implement custom Transformer.
#Kwargs help to give keyword arguments,@keyword_only  will store input keyword arguments

class BattingAverageTransform(
    Transformer,
    HasInputCols,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
	#Constructor: set values for all Param objects
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(BattingAverageTransform, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
        
	#The input passed to  _transform() is the entire input Dataset
    def _transform(self, dataset):
        input_cols = self.getInputCols()
        output_col = self.getOutputCol()
        dataset = dataset.withColumn(output_col, round(dataset[input_cols[0]] / dataset[input_cols[1]], 3))
        return dataset

batting_avg_transform = BattingAverageTransform(
	    inputCols=["Total_Hit", "Total_atBat"], outputCol="Batting_Average"
)
rolling_average = batting_avg_transform.transform(rolling_df)
rolling_average.show()

#create column and put into a single column for modeling -Categorical by encoding

test_df = spark.sql(
	        """
	        SELECT
	                *
	                , SPLIT(CONCAT(
	                    CASE WHEN batter IS NULL THEN ""
	                    ELSE batter END,
	                    " ",
	                    CASE WHEN game_id IS NULL THEN ""
	                    ELSE game_id END,
	                    " ",
	                    CASE WHEN atBat IS NULL THEN ""
	                    ELSE atBat END,
	                    " ",
	                    CASE WHEN Hit IS NULL THEN ""
	                    ELSE Hit END
	                ), " ") AS categorical
	            FROM batter_view
	        """
	    )
test_df.show()

#Count Vectorizer- to tranform data into vectors /Also enables preprocessing of data

count_vectorizer = CountVectorizer(
	inputCol="categorical", outputCol="categorical_vector"
)

count_vectorizer_fitted = count_vectorizer.fit(test_df)
test_df = count_vectorizer_fitted.transform(test_df)
test_df.show()

#Random Forest Classifier used to merge different decisions and make accurate prediction

random_forest = RandomForestClassifier(
labelCol="Hit",
featuresCol="categorical_vector",
numTrees=100,
predictionCol="will_hit",
probabilityCol="prob_of_hit",
rawPredictionCol="raw_pred_hit",
)

random_forest_fitted = random_forest.fit(test_df)
rolling_df = random_forest_fitted.transform(test_df)

#Final Output

rolling_df.select("game_id", "batter","Hit","atBat","categorical","categorical_vector").show()

    

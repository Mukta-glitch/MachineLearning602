import sys

from pyspark import StorageLevel, keyword_only
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import FeatureHasher
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, lit, split, when

import warnings
warnings.simplefilter(action='ignore')

appName = "Pyspark Homework 3"
master = "local"
# Create Spark session
spark = SparkSession.builder \
    .appName(appName) \
    .master(master) \
    .getOrCreate()

sql = "select * from baseball.batter_counts"
sql1="select * from baseball.game"
database = "baseball"
user = "admin_user"
password = "secret_password"
server = "localhost"
port = "3306"
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"

# Create a data frame by reading data from Mariadb via JDBC
batting_df = spark.read.format("jdbc") \
    .option("url", jdbc_url) \
    .option("query", sql) \
    .option("user", user) \
    .option("password", password) \
    .option("driver", jdbc_driver) \
    .load()

batting_df.select("batter", "game_id", "atBat", "Hit").show()
batting_df.createOrReplaceTempView("batter_view")
batting_df.persist(StorageLevel.MEMORY_ONLY)

rolling_df = spark.read.format("jdbc") \
    .option("url", jdbc_url) \
    .option("query", sql1) \
    .option("user", user) \
    .option("password", password) \
    .option("driver", jdbc_driver) \
    .load()
    
rolling_df.select("game_id", "local_date").show()
rolling_df.createOrReplaceTempView("game")
rolling_df.persist(StorageLevel.MEMORY_ONLY)

bat_avg = spark.sql(
	"""
	select
	    batter
	    , game_id
	    , atBat
	    , Hit 
	    , (Hit/nullif(atBat,0)) as game_bat_avg
	    , case when Hit > 0 then 1 else 0 end as ahit
	from batter_view
	order by batter, game_id
	"""
	)
bat_avg.show()

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

count_vectorizer = CountVectorizer(
	inputCol="categorical", outputCol="categorical_vector"
)

count_vectorizer_fitted = count_vectorizer.fit(test_df)
test_df = count_vectorizer_fitted.transform(test_df)
test_df.show()
    
random_forest = RandomForestClassifier(
labelCol="Hit",
featuresCol="categorical_vector",
numTrees=100,
predictionCol="will_hit",
probabilityCol="prob_of_hit",
rawPredictionCol="raw_pred_hit",
)

random_forest_fitted = random_forest.fit(test_df)
test_df = random_forest_fitted.transform(test_df)
test_df.select("game_id", "batter","Hit","atBat","categorical").show()
    

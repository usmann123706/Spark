# Regression Using Spark

## Objectives
After completing this lab you will be able to:

- Use PySpark to connect to a Spark cluster.
- Create a Spark session.
- Read a CSV file into a DataFrame.
- Split the dataset into training and testing sets.
- Use VectorAssembler to combine multiple columns into a single vector column.
- Use Linear Regression to build a prediction model.
- Use metrics to evaluate the model.
- Stop the Spark session.

## Installing Modules

!pip install pyspark==3.1.2 -q  
!pip install findspark -q  

# Regression Using Spark

# Installing Modules
!pip install pyspark==3.1.2 -q
!pip install findspark -q

# Importing Required Libraries
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Create a Spark Session
spark = SparkSession.builder.appName("Regressing using SparkML").getOrCreate()

# Load the Data
mpg_data = spark.read.csv("mpg.csv", header=True, inferSchema=True)
mpg_data.printSchema()
mpg_data.show(5)

# Identify the Label Column and the Input Columns
assembler = VectorAssembler(inputCols=["Cylinders", "Engine Disp", "Horsepower", "Weight", "Accelerate", "Year"],outputCol="features")
mpg_transformed_data = assembler.transform(mpg_data)
mpg_transformed_data.select("features", "MPG").show()

# Split the Data
(training_data, testing_Data) = mpg_transformed_data.randomSplit([0.7, 0.3], seed=42)

# Build and Train a Linear Regression Model
lr = LinearRegression(featuresCol="features", labelCol="MPG")
model = lr.fit(training_data)

# Evaluate the Model
predictions = model.transform(testing_data)
evaluator = RegressionEvaluator(labelCol="MPG", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(predictions)
print("R Squared =", r2)

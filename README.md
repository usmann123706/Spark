# Classification Using Spark

## Objectives

-Use PySpark to connect to a spark cluster.  
-Create a spark session.  
-Read a csv file into a data frame.  
-Split the dataset into training and testing sets.  
-Use VectorAssembler to combine multiple columns into a single vector column.  
-Use Logistic Regression to build a classification model.  
-Use metrics to evaluate the model.  
-Stop the spark session.  

## Installing Modules

!pip install pyspark==3.1.2 -q  
!pip install findspark -q  
  
# Importing Required Libraries

import findspark  
findspark.init()  
from pyspark.sql import SparkSession  
from pyspark.ml.feature import StringIndexer  
from pyspark.ml.feature import VectorAssembler  
from pyspark.ml.classification import LogisticRegression  
from pyspark.ml.evaluation import MulticlassClassificationEvaluator  

# Create a Spark Session
spark = SparkSession.builder.appName("Iris Flower Classification").getOrCreate()   

# Load the Data
iris_data = spark.read.csv("iris.csv", header=True, inferSchema=True)  
iris_data.show(5)    

# Identify the Label Column and the Input Columns
indexer = StringIndexer(inputCol='Species', outputCol='label')  
iris_data = indexer.fit(iris_data).transform(iris_data)  
iris_data.groupby('label','species').count().orderBy('label').show()   
assembler = VectorAssembler(inputCols=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"], outputCol='features')  
iris_transformed_data = assembler.transform(iris_data)  
iris_transformed_data.select("features","label").show(2000)  

# Split the Data
(training_data, testing_data) = iris_transformed_data.randomSplit([0.7,0.3], seed=42)    

# Build and Train a Logistic Regression Model
lr = LogisticRegression(featuresCol="features", labelCol="label")  
model = lr.fit(training_data)    

# Evaluate the Model
predictions = model.transform(testing_data)  
predictions.show()  
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")  
accuracy = evaluator.evaluate(predictions)  
print("Accuracy =", accuracy)  

# Make predictions using user input
input_data = [(5.7, 2.8, 4.1, 1.3)]
df = spark.createDataFrame(input_data, ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])  
assembler = VectorAssembler(inputCols=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"], outputCol='features')  
df_transformed = assembler.transform(df)  
prediction = model.transform(df_transformed)  
prediction.select("features", "prediction").show()  

# Clustering Using Spark

## Objectives

Use PySpark to connect to a spark cluster.  
Create a spark session.  
Read a csv file into a data frame.  
Use KMeans algorithm to cluster the data.  
Stop the spark session.  

## Installing Modules

!pip install pyspark==3.1.2 -q  
!pip install findspark -q  
  
# Importing Required Libraries

import findspark  
findspark.init()  
from pyspark.ml.clustering import KMeans  
from pyspark.ml.feature import VectorAssembler  
from pyspark.sql import SparkSession  

# Create a Spark Session
spark = SparkSession.builder.appName("Seed Clustering").getOrCreate()   

# Load the Data
seed_data =  spark.read.csv("seeds.csv", header=True, inferSchema=True)  
seed_data.printSchema()  
seed_data.show(5)  

# Create a feature vector
feature_cols = ['area',
 'perimeter',
 'compactness',
 'length of kernel',
 'width of kernel',
 'asymmetry coefficient',
 'length of kernel groove']  
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")  
seed_transformed_data =  assembler.transform(seed_data)  
seed_transformed_data.show(5)  

# Create a clustering model
number_of_clusters =  7  
kmeans = KMeans(k = number_of_clusters)  
model = kmeans.fit(seed_transformed_data)  

# Print Cluster Details
predictions =  model.transform(seed_transformed_data)  
predictions.select("features", "prediction").show(n=5, truncate=False, vertical=True)  
predictions.groupBy('prediction').count().show()  

# Predict using cluster details
input_data = [(13.99,13.83,0.9183,5.119,3.383,5.234,4.781)]  
input_df = spark.createDataFrame(input_data, ['area', 'perimeter', 'compactness', 'length of kernel', 'width of kernel', 'asymmetry coefficient','length of kernel groove'])  
input_assembler = VectorAssembler(inputCols=['area', 'perimeter', 'compactness', 'length of kernel', 'width of kernel', 'asymmetry coefficient','length of kernel groove'], outputCol="features")  
input_transformed = input_assembler.transform(input_df)  
predictions = model.transform(input_transformed)  
predictions.select("features", "prediction").show()  
spark.stop()  

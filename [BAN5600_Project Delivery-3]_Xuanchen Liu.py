# Databricks notebook source
# MAGIC %md
# MAGIC # Read dataset

# COMMAND ----------

# Import Dataset

from pyspark.sql import SparkSession

# Start a SparkSession
spark = SparkSession.builder.appName("MyApp").getOrCreate()

# Load data from a CSV file
Bank = spark.read.csv('/FileStore/tables/BAN5600_Project__dataset-1.csv', header=True, inferSchema=True)


# COMMAND ----------

# Print the schema of the dataset
Bank.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Statistic Summary of the dataset

# COMMAND ----------

# Get the number of observations in the DataFrame
num_obs = Bank.count()

# Print the number of observations
print("Number of observations:", num_obs)

# Get the number of variables in the DataFrame
num_vars = len(Bank.columns)

# Print the number of variables
print("Number of variables:", num_vars)

# COMMAND ----------

# Display 10 observations randomly selected without truncat
Bank.show(10, truncate=False)

# COMMAND ----------

# Statistical Summary of the dataset
summary = Bank.describe().toPandas()

# Print the summary
print(summary)

# COMMAND ----------

Bank = Bank.filter((Bank.balance >= 0)) # Filter out negative balance value
print(Bank.describe().toPandas()) # Now the min balance is 0

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Descritive Analytics and Visualiztion

# COMMAND ----------

Bank.registerTempTable("data")
display(sqlContext.sql("select * from data"))

# COMMAND ----------

# distribution of cities in database for each category of survival
Bank.registerTempTable("data")
display(sqlContext.sql("select * from data"))

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Data Preprocessing
# MAGIC ## 3.1 Data Cleaning
# MAGIC ## 3.2 Data Augementation
# MAGIC ## 3.3 Spliting dataset
# MAGIC ## 3.4 Feature Engineering

# COMMAND ----------

# 3.1 Data Cleaning
# 3.1.1 Checking Missing Values
from pyspark.sql.functions import * 
from pyspark.sql.functions import when, count, col
Bank.select([count(when(isnull(c), c)).alias(c) for c in Bank.columns]).show()
# This dataset does not contain any missing values


# COMMAND ----------

# 3.1.2 Checking duplicate values in the DataFrame
num_duplicates = Bank.count() - Bank.dropDuplicates().count()

# Print the number of duplicates
print("Number of duplicates:", num_duplicates)
# This dataset does not contain any duplcate values

# COMMAND ----------

# 3.1.3 Checking Outliers
from sklearn.impute import SimpleImputer
import pandas as pd

# Setting quantiles range
quantiles = {
    c: dict(
        zip(["q1", "q3"], Bank.approxQuantile(c, [0.25, 0.75], 0))
    )
    for c in ["balance", "duration","campaign"]
}
quantiles

# COMMAND ----------

# Calculating quantiles for each variable
for i in quantiles:
    iqr = quantiles[i]['q3'] - quantiles[i]['q1']
    quantiles[i]['lower_bound'] = quantiles[i]['q1'] - (iqr * 1.5)
    quantiles[i]['upper_bound'] = quantiles[i]['q3'] + (iqr * 1.5)
print(quantiles)

# Selecting outliers
import pyspark.sql.functions as f
Bank_clean=Bank.select(
    "*",
    *[
        f.when(
            f.col(c).between(quantiles[c]['lower_bound'], quantiles[c]['upper_bound']),
            0
        ).otherwise(1).alias(c+"_out") 
        for c in ["balance", "duration","campaign"]
    ]
)
Bank_clean.show(10, truncate=False)

# COMMAND ----------

from pyspark.sql.functions import col
Bank_clean=Bank_clean.withColumn("outliers", col("balance_out")+col("duration_out")+col("campaign_out"))
Bank_clean.show(10)

# COMMAND ----------

# Dropping outliers
Bank_clean = Bank_clean.filter((Bank_clean.outliers == 0) )
Bank_clean=Bank_clean.select(["balance", "duration","education","campaign", "contact", "default", "housing", "marital", "y"])
Bank_clean.select("balance", "duration","campaign").describe().show(10, truncate=False)

# COMMAND ----------

import numpy as np
print("proportion of the lost Rows: ",np.round((Bank.count()-Bank_clean.count())/Bank.count(),4))
# After dropping outliers, we missed 22.15% rows

# COMMAND ----------

# we exclude unrelevant columns
Bank_clean = Bank_clean.select(['marital',  'education',  'housing',  'balance',  'contact',  'duration', 'campaign',
 'y'])

Bank_clean.show() # Now the dataset only contains 8 columns

# COMMAND ----------

# 3.2 Data Augmentation
# Target variable bias validating

# 3.2.1 Checking if the target variable is unbalanced
display(Bank_clean
        .groupBy("y")
        .count()
        .sort("count", ascending=False)) # The target variable is unbalanced

# COMMAND ----------

# 3.2.2 Oversampling the unblanced variable

#spliting the dataset by classes
major_df = Bank_clean.filter(col('y') == 'no')
minor_df = Bank_clean.filter(col('y') == 'yes')

#ratio of number observation major vs minor class
r = int(major_df.count()/minor_df.count())

spark.conf.set("spark.sql.shuffle.partitions", "1") # Setting the partitions consistent

# duplicate the minority rows
oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in range(r)]))).drop('dummy')
 
# combine both oversampled minority rows and previous majority rows 
combined_Bank_clean = major_df.unionAll(oversampled_df)
Bank_clean1 = combined_Bank_clean
                                                           
import matplotlib.pyplot as plt
df = Bank_clean1.groupBy('y').count().toPandas()
ax = df.plot.bar(x='y', rot=0, title='Number of Observations after Oversampling', label = False)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2., p.get_height(), int(p.get_height()), 
            fontsize=10, color='black', ha='center', va='bottom')
plt.show()

# COMMAND ----------

# 3.3 Spliting the dataset to training set(70%) and the test set(30%)
train_bank_data, test_bank_data = Bank_clean1.randomSplit([0.7, 0.3], seed=42)

print(train_bank_data.cache().count()) # Cache because accessing training data multiple times
print(test_bank_data.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 Feature Engineering

# COMMAND ----------

# 3.4 Feature Engineering


# 3.4.1 Convert categorical variables to numerical variables
from pyspark.ml.feature import StringIndexer, OneHotEncoder

# Define the categorical columns
categorical_cols = ['marital', 'education', 'housing', 'contact']
 
# The following two lines are estimators. They return functions that we will later apply to transform the dataset.
# Convert the column of string values to a column of label indexes. 
stringIndexer = StringIndexer(inputCols=categorical_cols, outputCols=[x + "Index" for x in categorical_cols]) 
# Map the column of category indices to the column of binary vectors
encoder = OneHotEncoder(inputCols=stringIndexer.getOutputCols(), outputCols=[x + "OHE" for x in categorical_cols]) 
 
# The label column ("y") is also a string value - it has two possible values, "yes" and "no". 
# Convert it to a numeric value using StringIndexer.
labelToIndex = StringIndexer(inputCol="y", outputCol="label")

# COMMAND ----------

# Checking the dataset after converting string to numerical
stringIndexerModel = stringIndexer.fit(train_bank_data)
display(stringIndexerModel.transform(train_bank_data))
stringIndexerModel.transform(train_bank_data).printSchema()

# COMMAND ----------

# 3.4.2 Combine all feature columns into a single feature vector
from pyspark.ml.feature import VectorAssembler
 
# Define numerical columns
numerical_cols = ['balance', 'duration', 'campaign']

# This includes both the numeric columns and the one-hot encoded binary vector columns in our dataset.
assemblerInputs = [c + "OHE" for c in categorical_cols] + numerical_cols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Predictive Analytics
# MAGIC ## 4.1 Logistic Regression Model 
# MAGIC ### (Including Comapring the model accuracy without oversampling and the model accuracy after oversampling)
# MAGIC ## 4.2 Decision Tree Model (Classification Tree)
# MAGIC ## 4.3 Gradient Boost Tree Model

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Logistice Regression Model

# COMMAND ----------

# 4.1.1 Define the Logistic Regression model
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol="features", labelCol="label", regParam=1.0)

# Define a pipeline to automate and ensure repeatability of the transformations to be applied to the dataset
from pyspark.ml import Pipeline
 
# 4.1.2 Define the pipeline based on the stages created in previous steps.
pipeline = Pipeline(stages=[stringIndexer, encoder, labelToIndex, vecAssembler, lr])
 
# 4.1.3 Define the pipeline model.
lr_pipelineModel = pipeline.fit(train_bank_data)
 
# 4.1.4 Apply the pipeline model to the test dataset.
lr_predDF = lr_pipelineModel.transform(test_bank_data)
lr_predDF_train = lr_pipelineModel.transform(train_bank_data)

# Display the predictions from the model.
display(lr_predDF.select('features', 'label', 'prediction', 'probability'))
                           

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 4.1.5 Evaluate the ROC curve
bcEvaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print(f"Logistic Regression Model Area under ROC curve: {bcEvaluator.evaluate(lr_predDF)}")
print(f"Logistic Regression Model(train) Area under ROC curve: {bcEvaluator.evaluate(lr_predDF_train)}")

# 4.1.6 Evaluate the model accuracy
mcEvaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print(f"Logistice Regression Model Accuracy: {mcEvaluator.evaluate(lr_predDF)}")
print(f"Logistice Regression Model Training Accuracy: {mcEvaluator.evaluate(lr_predDF_train)}")


# COMMAND ----------

# 4.1.7 Hyperparameter tuning
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
 
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .build())

# Create a 5-fold CrossValidator
lr_cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=bcEvaluator, numFolds=5)

# Fit the cross-validator to the training data
lr_cvModel = lr_cv.fit(train_bank_data)

# Get the best model
lr_bestModel = lr_cvModel.bestModel

# Evaluate the best model on the test data
lrBest_results = lr_bestModel.transform(test_bank_data)
lrBest_results_train = lr_bestModel.transform(train_bank_data)


# COMMAND ----------

# 4.1.8 Evaluate the best model's performance based on area under the ROC curve and accuracy 
print(f"Logistic Reg Best Model Area under ROC curve: {bcEvaluator.evaluate(lrBest_results)}")
print(f"Logistic Reg Best Model Accuracy: {mcEvaluator.evaluate(lrBest_results)}") # Accuracy got improved
print(f"Logistic Reg Best Model Training Accuracy: {mcEvaluator.evaluate(lrBest_results_train)}")

display(lr_bestModel.stages[-1],  lrBest_results.drop("prediction", "rawPrediction", "probability"),"ROC")

# COMMAND ----------

# 4.1.9 Extracting the feature coefficients
import pandas as pd
from pyspark.ml.classification import LogisticRegressionModel

# Get the LogisticRegressionModel object from the best model
lrModel = lr_bestModel.stages[-1]

# Get the coefficients and feature names as a list of tuples
coefficients = lrModel.coefficients.toArray()
feature_names = vecAssembler.getInputCols()
coefficients_list = list(zip(feature_names, coefficients))

# Create a Pandas DataFrame and sort by coefficient values
df = pd.DataFrame(coefficients_list, columns=['Feature', 'Coefficient'])
df_sorted = df.sort_values(by='Coefficient', ascending=False)

# Print the sorted DataFrame
print(df_sorted)
# Print the model's intercept
print(f"Intercept: {lrModel.intercept}")


# COMMAND ----------

# 4.1.10 Evaluate the logistice Regression model
display(lr_bestModel.stages[-1],  lrBest_results.drop("prediction", "rawPrediction", "probability"),"ROC")

# COMMAND ----------

# 4.1.11 Data visualization for the prediction result
lrBest_results.createOrReplaceTempView("finalPredictions")

# COMMAND ----------

# Data visualization for the prediction result

display(sqlContext.sql("select * from finalPredictions"))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT contact, prediction, label, y, count(*) AS count
# MAGIC FROM finalPredictions
# MAGIC GROUP BY contact, prediction, label, y
# MAGIC ORDER BY contact

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1.12 Comparing the accuracy of logistic model without oversampling and after oversampling

# COMMAND ----------

# Perform the logistic regression without oversampling
train_bank_bfoversampling, test_bank_bfoversampling = Bank_clean.randomSplit([0.7, 0.3], seed=42)
print(train_bank_bfoversampling.cache().count()) # Cache because accessing training data multiple times
print(test_bank_bfoversampling.count())

#Define the Logistic Regression model
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol="features", labelCol="label", regParam=1.0)

# Define a pipeline to automate and ensure repeatability of the transformations to be applied to the dataset
from pyspark.ml import Pipeline
 
# Define the pipeline based on the stages created in previous steps.
pipeline = Pipeline(stages=[stringIndexer, encoder, labelToIndex, vecAssembler, lr])
 
# Define the pipeline model.
lr_pipelineModel_bf = pipeline.fit(train_bank_bfoversampling)
 
# Apply the pipeline model to the test dataset.
lr_predDF_bf = lr_pipelineModel_bf.transform(test_bank_bfoversampling)

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
# Evaluate the ROC curve
bcEvaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print(f"Logistic Regression Model without oversampling Area under ROC curve: {bcEvaluator.evaluate(lr_predDF_bf)}")
 
# Evaluate the model accuracy
mcEvaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print(f"Logistice Regression Model without oversampling Accuracy: {mcEvaluator.evaluate(lr_predDF_bf)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Decision Tree Model (Classification Tree)

# COMMAND ----------

#Define the Classification Tree model
from pyspark.ml.classification import DecisionTreeClassifier
ct= DecisionTreeClassifier(featuresCol='features',labelCol='label')

# Define a pipeline to automate and ensure repeatability of the transformations to be applied to the dataset
from pyspark.ml import Pipeline
 
# Define the pipeline based on the stages created in previous steps.
ct_pipeline = Pipeline(stages=[stringIndexer, encoder, labelToIndex, vecAssembler, ct])
 
# Define the pipeline model.
ct_pipelineModel = ct_pipeline.fit(train_bank_data)
 
# Apply the pipeline model to the test dataset.
ct_predDF = ct_pipelineModel.transform(test_bank_data)
ct_predDF_train = ct_pipelineModel.transform(train_bank_data)

# Display the predictions from the model.
display(ct_predDF.select('features', 'label', 'prediction', 'probability'))

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Evaluate the ROC curve
bcEvaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print(f"Classification Tree Model Area under ROC curve: {bcEvaluator.evaluate(ct_predDF)}")
print(f"Classification Tree Trainging Model Area under ROC curve: {bcEvaluator.evaluate(ct_predDF_train)}")
  
# Evaluate the model accuracy
mcEvaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print(f"Classification Tree Model Accuracy: {mcEvaluator.evaluate(ct_predDF)}")
print(f"Classification Tree Training Model Accuracy: {mcEvaluator.evaluate(ct_predDF_train)}")

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Hyperparameter tuning

# Define the parameter grid
paramGrid1 = ParamGridBuilder() \
    .addGrid(ct.maxDepth, range(1, 10)) \
    .build()

paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .build())

# Create a 5-fold CrossValidator
ct_cv = CrossValidator(estimator=ct_pipeline, estimatorParamMaps=paramGrid1, evaluator=bcEvaluator, numFolds=5)

# Fit the cross-validator to the training data
ct_cvModel = ct_cv.fit(train_bank_data)

# Get the best model
ct_bestModel = ct_cvModel.bestModel

# Evaluate the best model on the test data
ctBest_results = ct_bestModel.transform(test_bank_data)
ctBest_results_train = ct_bestModel.transform(train_bank_data)

# COMMAND ----------

# Evaluate the model's performance based on area under the ROC curve and accuracy 
print(f"Classification Tree Best Model Area under ROC curve: {bcEvaluator.evaluate(ctBest_results)}")
print(f"Classification Tree Best Model Accuracy: {mcEvaluator.evaluate(ctBest_results)}") # Accuracy got improved
print(f"Classification Tree Best Model Training Accuracy: {mcEvaluator.evaluate(ctBest_results_train)}") # Accuracy got improved

display(ct_bestModel.stages[-1],  ctBest_results.drop("prediction", "rawPrediction", "probability"))

# COMMAND ----------

# Find the best Max depth
best_max_depth = ct_bestModel.stages[-1].getOrDefault('maxDepth')
print(best_max_depth)

importances = ct_bestModel.stages[-1].featureImportances
print(importances)

# COMMAND ----------

ct_stage = ct_bestModel.stages[-2]
input_cols = ct_stage.getInputCols()

print(input_cols)

# COMMAND ----------

# Extracting Feature Importance
from pyspark.ml.feature import OneHotEncoderModel, VectorAssembler

if isinstance(ct_stage, OneHotEncoderModel):
    feature_importances = [
        importances[input_cols.index(col)] * ct_stage.coefficients[col] 
        for col in input_cols
    ]
elif isinstance(ct_stage, VectorAssembler):
    input_cols = ct_stage.getInputCols()
    feature_importances = [
        importances[i] 
        for i in range(len(input_cols))
    ]
else:
    feature_importances = None  # handle other types of feature transformation stages here
    
    
feature_importance_dict = dict(zip(input_cols, feature_importances))

print(feature_importances)
print(input_cols)

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# compute feature importances
feature_importances_Insert = [0.005710027892963268, 0.008179158444561635, 0.004185895788902695, 0.006634751157811564, 0.015112235138786765, 0.11392097675599594, 0.01836947191345851]
input_cols_Insert = ['maritalOHE', 'educationOHE', 'housingOHE', 'contactOHE', 'balance', 'duration', 'campaign']

# create schema for dataframe
schema = StructType([StructField('Feature', StringType(), True), 
                     StructField('Importance', FloatType(), True)])

# create list of rows for dataframe
rows = [(col, imp) for col, imp in zip(input_cols_Insert, feature_importances_Insert)]

# create dataframe from list of rows and schema
df_importances = spark.createDataFrame(rows, schema=schema)

# sort dataframe by importance in descending order
df_importances = df_importances.orderBy(col('Importance').desc())
df_importances.show()


# COMMAND ----------

# Data visualization for the prediction result
ctBest_results.createOrReplaceTempView("ct_finalPredictions")
display(sqlContext.sql("select * from ct_finalPredictions"))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT campaign, prediction, label, count(*) AS count
# MAGIC FROM ct_finalPredictions
# MAGIC GROUP BY campaign, prediction, label
# MAGIC ORDER BY campaign

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 Gradient Boost Tree model

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(labelCol='label', featuresCol='features')
gbt_pipeline = Pipeline(stages=[stringIndexer, encoder, labelToIndex, vecAssembler, gbt])

# Define the pipeline model.
gbt_pipelineModel = gbt_pipeline.fit(train_bank_data)
 
# Apply the pipeline model to the test dataset.
gbt_predDF = gbt_pipelineModel.transform(test_bank_data)
gbt_predDF_train = gbt_pipelineModel.transform(train_bank_data)

# Display the predictions from the model.
display(gbt_predDF.select('features', 'label', 'prediction', 'probability'))


# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Evaluate the ROC curve
bcEvaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print(f"gb tree Model Area under ROC curve: {bcEvaluator.evaluate(gbt_predDF)}")
print(f"gb tree Training Model Area under ROC curve: {bcEvaluator.evaluate(gbt_predDF_train)}")

# Evaluate the model accuracy
mcEvaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print(f"gb tree Model Accuracy: {mcEvaluator.evaluate(gbt_predDF)}")
print(f"gb tree Training Model Accuracy: {mcEvaluator.evaluate(gbt_predDF_train)}")

# COMMAND ----------

# Hyperparameter Tuning
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid2 = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [2, 4, 6])
             .addGrid(gbt.maxBins, [20, 60])
             .addGrid(gbt.maxIter, [10, 20])
             .build())

gbt_cv = CrossValidator(estimator=gbt_pipeline, estimatorParamMaps=paramGrid2, evaluator=bcEvaluator, numFolds=5)

# Run cross validations to get the best model
gbt_cvModel = gbt_cv.fit(train_bank_data)
gbt_bestModel = gbt_cvModel.bestModel
gbtBest_results = gbt_bestModel.transform(test_bank_data)
gbtBest_results_train = gbt_bestModel.transform(train_bank_data)

# COMMAND ----------

# Evaluate the model's performance based on area under the ROC curve and accuracy 
print(f"gb tree Best Model Area under ROC curve: {bcEvaluator.evaluate(gbtBest_results)}")
print(f"gb tree Best Model Accuracy: {mcEvaluator.evaluate(gbtBest_results)}") # Accuracy got improved
print(f"gb tree Best Training Model Accuracy: {mcEvaluator.evaluate(gbtBest_results_train)}")

# COMMAND ----------

# Finding the best max depth
best_max_depth2 = gbt_bestModel.stages[-1].getOrDefault('maxDepth')
print(best_max_depth2)

importances2 = gbt_bestModel.stages[-1].featureImportances
print(importances2)

# COMMAND ----------

# Extracting feature importance
importances2 = gbt_bestModel.stages[-1].featureImportances
gbt_stage = gbt_bestModel.stages[-2]
input_cols2 = gbt_stage.getInputCols()

from pyspark.ml.feature import OneHotEncoderModel, VectorAssembler

if isinstance(gbt_stage, OneHotEncoderModel):
    feature_importances2 = [
        importances2[input_cols2.index(col)] * gbt_stage.coefficients[col] 
        for col in input_cols2
    ]
elif isinstance(gbt_stage, VectorAssembler):
    input_cols2 = gbt_stage.getInputCols()
    feature_importances2 = [
        importances2[i] 
        for i in range(len(input_cols2))
    ]
else:
    feature_importances2 = None  # handle other types of feature transformation stages here
    
    
feature_importance_dict2 = dict(zip(input_cols2, feature_importances2))

print(feature_importances2)
print(input_cols2)

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# compute feature importances
feature_importances_Insert2 = [0.0227809104260567, 0.01585924683118732, 0.017215588061929233, 0.030943186114886917, 0.03723224218439401, 0.06753674381986144, 0.013418849191545628]
input_cols_Insert2 = ['maritalOHE', 'educationOHE', 'housingOHE', 'contactOHE', 'balance', 'duration', 'campaign']

# create schema for dataframe
schema = StructType([StructField('Feature', StringType(), True), 
                     StructField('Importance', FloatType(), True)])

# create list of rows for dataframe
rows = [(col, imp) for col, imp in zip(input_cols_Insert2, feature_importances_Insert2)]

# create dataframe from list of rows and schema
df_importances2 = spark.createDataFrame(rows, schema=schema)

# sort dataframe by importance in descending order
df_importances2 = df_importances2.orderBy(col('Importance').desc())
df_importances2.show()


# COMMAND ----------

# Data visualization for the prediction result
gbtBest_results.createOrReplaceTempView("gbt_finalPredictions")
display(sqlContext.sql("select * from gbt_finalPredictions"))

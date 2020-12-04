from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

if __name__ == "__main__":
    spark = SparkSession.builder.appName("ALSExample").getOrCreate

    spark.sparkContext.setLogLevel("ERROR")

    lines = spark.read.text("s3://orderlogs-swarupmishal/2020/09/18/01/*").rdd
    parts = lines.map(lambda row: row.value.split(','))
    # Filter out postage, shipping, bank charges, discounts, commissions
    productsOnly = parts.filter(lambda p: p[1][0:5].isdigit())
    # Filter out empty customer ID's
    cleanData = productsOnly.filter(lambda p: p[6].isdigit())
    ratingsRDD = cleanData.map(lambda p: Row(customerId=int(p[6]), itemId = int(p[1][0:5]), rating=1.0))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2])

    # Build the recommendation model using ALS on the training data
    # Note we set cold strat strategy to 'drop' to ensure we don't get NaN
    als = ALS(maxIter=5, regParam=0.01, userCol="customerId", itemCol="itemId", coldStartStrategy="drop")
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    # Generate top 10 item recommendations for each Customers
    customerRecs = model.recommendForAllUsers(10)

    # Generate top 10 item recommendations for a specified set of users
    customers = ratings.select(als.getUserCol()).distinct().limit(3)
    customerSubsetRecs = model.recommendForUserSubset(customers, 10)

    customerRecs.show()
    customerSubsetRecs.show()

    spark.stop()


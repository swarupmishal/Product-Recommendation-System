# Product-Recommendation-System
![alt text](https://github.com/swarupmishal/Product-Recommendation-System/blob/main/extras/overview.png)

## Architecture and Overview:
![alt text](https://github.com/swarupmishal/Product-Recommendation-System/blob/main/extras/architecture.png)

Built a Product Recommendation System using historical order logs of an online retail store. The data was generated on EC2 instance where Kinesis agent was installed, and published through Kinesis data firehose into an Amazon S3 data lake. The file "log_generator.py" was used for generating log files. An Elastic MapReduce (EMR) cluster was spinned up to pick up these files from the S3 data lake. An Apache Spark MLlib job was running on the EMR cluster to produce a recommendations model. Please take a peek at the code "als_product_recommendation.py" which performs this task. The ML job predicted what items the customers would like to buy based on the collaborative filtering model.
![alt text](https://github.com/swarupmishal/Product-Recommendation-System/blob/main/extras/collaborative%20filtering.png)

### Data:
The data is from a UK retailer that sells crafty goods available in the public domain. Its an ecommerce dataset with real world problems.

### Cloud Provider:
AWS
### Environment: 
Spark 2.4.0 on Hadoop 2.8.5 Yarn with Zeppelin 0.8.0
### EMR Cluster: 
1 Master and 2 Core nodes
### Algorithm: 
Alternating Least Squares (ALS) from pyspark module


import pyspark
import json
from datetime import datetime
import pandas as pd

import findspark
findspark.init()

from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.pipeline import PipelineModel

spark = SparkSession.builder.appName('ALS').getOrCreate()

def mode_als(df):
    # read data
    df = df[["customer_id", "product_id", "rating"]].dropna()
    df["customer_id"] = df["customer_id"].astype(int)
    df["product_id"] = df["product_id"].astype(int)
    df["rating"] = df["rating"].astype(int)
    data_train = spark.createDataFrame(df)
    data_train = data_train.dropDuplicates()

    # split data 
    train, test = data_train.randomSplit([0.9, 0.1])
    # model
    als = ALS(rank = 10,
            regParam = 0.5,
            userCol="customer_id",
            itemCol="product_id",
            ratingCol="rating",
            maxIter = 10,
            coldStartStrategy="drop",
            nonnegative=True)

    # train model
    model = als.fit(train)
    results = model.transform(test)
    # evaluate model
    evaluator = RegressionEvaluator(metricName="rmse",
                                    labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(results)
    print("RMSE =",rmse, "của tập test")
    
    # save
    recommendations = model.recommendForAllUsers(12)
    df_result = recommendations.toPandas()
    test_dit = df_result.set_index('customer_id')['recommendations'].to_dict()
    dict_save = {}
    for ele in test_dit.keys():
       dict_save[str(ele)] = [(x['product_id'],x['rating']) for x in test_dit[ele]]
    date = int(datetime.now().timestamp())
    with open(f'data/mode_als_{date}.json', 'w') as fp:
        json.dump(dict_save, fp)
        
    return rmse 

if __name__ =="__main__":
        path = "data/ReviewRaw.csv"
        df = pd.read_csv(path)
        results = mode_als(df)
        print("Done")
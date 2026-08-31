"""Concrete-strength regression with Spark feature engineering and scaling."""

from __future__ import annotations

import argparse

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import SQLTransformer, StandardScaler, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession


SCHEMA = """cement double, blast_furnace_slag double, fly_ash double, water double,
superplasticizer double, coarse_aggregate double, fine_aggregate double, age int,
compressive_strength double"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("concrete-strength-regression").getOrCreate()
    train = spark.read.csv(args.train, schema=SCHEMA)
    test = spark.read.csv(args.test, schema=SCHEMA)
    age_bucket = SQLTransformer(
        statement="SELECT *, CASE WHEN age < 28 THEN 0 WHEN age < 90 THEN 1 ELSE 2 END AS age_band FROM __THIS__"
    )
    features = [column for column in train.columns if column != "compressive_strength"] + ["age_band"]
    assembler = VectorAssembler(inputCols=features, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features")
    model = RandomForestRegressor(
        labelCol="compressive_strength", featuresCol="features", numTrees=100, seed=1234
    )
    fitted = Pipeline(stages=[age_bucket, assembler, scaler, model]).fit(train)
    predictions = fitted.transform(test)
    rmse = RegressionEvaluator(labelCol="compressive_strength", metricName="rmse").evaluate(predictions)
    print(f"test_rmse={rmse:.4f}")
    spark.stop()


if __name__ == "__main__":
    main()


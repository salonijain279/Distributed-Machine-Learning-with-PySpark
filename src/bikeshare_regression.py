"""Gradient-boosted demand forecasting for hourly bike-share rentals."""

from __future__ import annotations

import argparse

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession


EXCLUDED = {"instant", "dteday", "casual", "registered", "cnt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("bikeshare-gbt-regression").getOrCreate()
    data = spark.read.option("header", True).option("inferSchema", True).csv(args.input)
    feature_columns = [column for column in data.columns if column not in EXCLUDED]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")
    indexer = VectorIndexer(inputCol="raw_features", outputCol="features", maxCategories=24)
    gbt = GBTRegressor(labelCol="cnt", featuresCol="features", seed=1234)
    pipeline = Pipeline(stages=[assembler, indexer, gbt])
    train, test = data.randomSplit([0.8, 0.2], seed=1234)
    evaluator = RegressionEvaluator(labelCol="cnt", metricName="rmse")
    grid = ParamGridBuilder().addGrid(gbt.maxDepth, [3, 5, 8]).addGrid(gbt.maxIter, [30, 60]).build()
    fitted = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=3,
    ).fit(train)
    rmse = evaluator.evaluate(fitted.transform(test))
    print(f"test_rmse={rmse:.4f}")
    spark.stop()


if __name__ == "__main__":
    main()


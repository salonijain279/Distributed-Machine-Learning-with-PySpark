"""Decision-tree classification for a categorical used-car evaluation dataset."""

from __future__ import annotations

import argparse

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession


FEATURES = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
SCHEMA = ",".join([f"{name} string" for name in FEATURES] + ["acceptability string"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("used-car-decision-tree").getOrCreate()
    train = spark.read.csv(args.train, schema=SCHEMA)
    test = spark.read.csv(args.test, schema=SCHEMA)
    indexers = [
        StringIndexer(inputCol=column, outputCol=f"{column}_idx", handleInvalid="keep")
        for column in FEATURES + ["acceptability"]
    ]
    assembler = VectorAssembler(
        inputCols=[f"{column}_idx" for column in FEATURES], outputCol="features"
    )
    tree = DecisionTreeClassifier(labelCol="acceptability_idx", featuresCol="features", seed=1234)
    model = Pipeline(stages=[*indexers, assembler, tree]).fit(train)
    predictions = model.transform(test)
    accuracy = MulticlassClassificationEvaluator(
        labelCol="acceptability_idx", metricName="accuracy"
    ).evaluate(predictions)
    print(f"test_accuracy={accuracy:.4f}")
    spark.stop()


if __name__ == "__main__":
    main()

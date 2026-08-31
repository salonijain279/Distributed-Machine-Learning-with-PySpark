"""Random-Forest survival classification with a fitted Spark ML pipeline."""

from __future__ import annotations

import argparse

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession


CATEGORICAL = ["Sex", "Embarked"]
NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]


def build_pipeline() -> Pipeline:
    imputer = Imputer(inputCols=["Age", "Fare"], outputCols=["Age_filled", "Fare_filled"])
    indexers = [
        StringIndexer(inputCol=column, outputCol=f"{column}_idx", handleInvalid="keep")
        for column in CATEGORICAL
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{column}_idx" for column in CATEGORICAL],
        outputCols=[f"{column}_ohe" for column in CATEGORICAL],
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=["Pclass", "Age_filled", "SibSp", "Parch", "Fare_filled"]
        + [f"{column}_ohe" for column in CATEGORICAL],
        outputCol="features",
    )
    model = RandomForestClassifier(labelCol="Survived", featuresCol="features", seed=1234)
    return Pipeline(stages=[imputer, *indexers, encoder, assembler, model])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("titanic-spark-classifier").getOrCreate()
    data = spark.read.option("header", True).option("inferSchema", True).csv(args.input)
    train, test = data.randomSplit([0.8, 0.2], seed=1234)
    pipeline = build_pipeline()
    forest = pipeline.getStages()[-1]
    grid = (
        ParamGridBuilder()
        .addGrid(forest.numTrees, [50, 100])
        .addGrid(forest.maxDepth, [4, 8])
        .build()
    )
    evaluator = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderROC")
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, numFolds=3)
    fitted = cv.fit(train)
    auc = evaluator.evaluate(fitted.transform(test))
    print(f"test_auc={auc:.4f}")
    spark.stop()


if __name__ == "__main__":
    main()


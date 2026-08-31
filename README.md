# Distributed Machine Learning with PySpark

Four end-to-end Spark ML projects covering classification, regression, feature engineering, evaluation, and cross-validated model selection. Each pipeline is written as a portable `spark-submit` job instead of a course-export notebook.

## Casebook

| Project | Analytics task | Pipeline |
|---|---|---|
| Titanic survival | Binary classification | Missing-value handling, categorical encoding, Random Forest, AUC evaluation |
| Bike-share demand | Regression | Leakage-safe feature selection, vector indexing, Gradient-Boosted Trees, RMSE tuning |
| Used-car acceptability | Multiclass classification | String indexing, vector assembly, Decision Tree, accuracy evaluation |
| Concrete strength | Regression | Age bucketing, scaling, Random Forest regression, RMSE evaluation |

## Why Spark ML

Spark ML keeps data preparation and modelling stages in one fitted pipeline, reducing training-serving skew. This repository demonstrates the same pattern across datasets with different target types and feature structures.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

spark-submit src/titanic_classifier.py --input data/titanic.csv
spark-submit src/bikeshare_regression.py --input data/hour.csv
spark-submit src/used_car_classifier.py --train data/cars_train.csv --test data/cars_test.csv
spark-submit src/concrete_regression.py --train data/concrete_train.csv --test data/concrete_test.csv
```

Every command prints an out-of-sample metric and the selected model configuration. Large/public datasets are not duplicated in the repository.

## Skills represented

`PySpark` · `Spark MLlib` · `Pipeline` · `StringIndexer` · `OneHotEncoder` · `Imputer` · `VectorAssembler` · `CrossValidator` · `Random Forest` · `Gradient-Boosted Trees`

## Origin

Rebuilt from MSBA Big Data Analytics labs as an original, reusable portfolio casebook. Exam material and instructor solution exports are excluded.


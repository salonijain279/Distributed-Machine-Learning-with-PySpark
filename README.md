# SparkPredict: PySpark ML Pipelines

Four portable Spark ML pipelines for binary and multiclass classification and regression, each with feature engineering, out-of-sample evaluation, and cross-validated model selection. Every pipeline runs as a standalone `spark-submit` job with explicit inputs, so the same code scales from a laptop to a cluster.

## Pipelines

| Pipeline | Task | Stages |
|---|---|---|
| Titanic survival | Binary classification | Missing-value handling, categorical encoding, Random Forest, AUC evaluation |
| Bike-share demand | Regression | Leakage-safe feature selection, vector indexing, Gradient-Boosted Trees, RMSE tuning |
| Used-car acceptability | Multiclass classification | String indexing, vector assembly, Decision Tree, accuracy evaluation |
| Concrete strength | Regression | Age bucketing, scaling, Random Forest regression, RMSE evaluation |

## Pipeline design

```mermaid
flowchart LR
    A[Raw data] --> B[Impute / encode / assemble]
    B --> C[Fitted Spark ML Pipeline]
    C --> D[CrossValidator model selection]
    D --> E[Out-of-sample metric]
```

Spark ML pipelines keep data preparation and modelling stages inside the same fitted workflow, reducing the risk of inconsistent transformations between training and scoring. The four implementations apply that pattern across different target types and evaluation metrics.

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

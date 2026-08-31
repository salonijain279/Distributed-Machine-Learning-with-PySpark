# Distributed Machine Learning with PySpark

I rebuilt four Spark ML coursework exercises as standalone projects covering classification, regression, feature engineering, evaluation, and cross-validated model selection. I converted the original notebook-style work into portable `spark-submit` jobs so the pipelines are easier to inspect and rerun.

## What I built

| Project | Analytics task | Pipeline |
|---|---|---|
| Titanic survival | Binary classification | Missing-value handling, categorical encoding, Random Forest, AUC evaluation |
| Bike-share demand | Regression | Leakage-safe feature selection, vector indexing, Gradient-Boosted Trees, RMSE tuning |
| Used-car acceptability | Multiclass classification | String indexing, vector assembly, Decision Tree, accuracy evaluation |
| Concrete strength | Regression | Age bucketing, scaling, Random Forest regression, RMSE evaluation |

## Why Spark ML

I used Spark ML pipelines to keep data preparation and modelling stages inside the same fitted workflow. Applying that pattern across different target types helped me practice avoiding inconsistent transformations between training and scoring.

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

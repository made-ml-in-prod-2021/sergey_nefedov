Homework1
==============================

Create environment
~~~
conda create -n $env_name python=3.6
conda activate $env_name
pip install -e .
~~~

Train model (RandomForestClassifier):
~~~
python src/train_pipeline.py configs/train_config_rf.yaml
~~~

Train model (LogisticRegression):
~~~
python src/train_pipeline.py configs/train_config_logreg.yaml
~~~

Predict:
~~~
python src/predict_pipeline.py configs/predict_config.yaml
~~~

Test:
~~~
pytest tests/
~~~

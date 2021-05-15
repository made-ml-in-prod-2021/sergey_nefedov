Homework2
==============================
# python:
Run app:
~~~
python app.py
~~~
Run request script:
~~~
python make_request.py
~~~


# Docker:
Build:
~~~
docker build -t sergeynefedov/inference:v3 .
~~~
Run:
~~~
docker run -p 8000:8000 sergeynefedov/inference:v3
~~~
Push:
~~~
docker login
docker push sergeynefedov/inference:v3
~~~
Pull command:
~~~
docker pull sergeynefedov/inference:v3
~~~
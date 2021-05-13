FROM python:3.6-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY configs/ ./configs/
COPY models/model.pkl ./models/model.pkl
COPY src/entities/ ./src/entities/

COPY app.py ./app.py

CMD ["python", "app.py"]
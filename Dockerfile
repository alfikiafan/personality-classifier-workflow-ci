FROM python:3.12-slim

WORKDIR /app

COPY MLProject/ .

RUN pip install --upgrade pip && \
    pip install mlflow pandas scikit-learn matplotlib seaborn shap joblib

ENTRYPOINT ["python", "modelling.py"]
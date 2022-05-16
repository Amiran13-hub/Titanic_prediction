
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY ./app /app

WORKDIR /app

RUN pip install sklearn joblib

CMD ["uvicorn", "maintitanic:app", "--host", "0.0.0.0", "--port", "80"]


FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

WORKDIR /app/service

COPY ./ifcb_features_service .
COPY ./pyproject.toml .
COPY ./README.md .

RUN pip install .

EXPOSE 8010

WORKDIR /app

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8010"]
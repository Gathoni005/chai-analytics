FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG CACHEBUST=2
COPY . /app/

RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 5000

ENV FLASK_ENV=production
ENV FLASK_DEBUG=0
ENV FLASK_APP=app.py

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "2", "app:app"]
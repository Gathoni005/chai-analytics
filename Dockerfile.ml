FROM pytorch/pytorch:latest


WORKDIR /app

# System deps (needed for some scipy/statsmodels wheels on slim)
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    build-essential \
#    gcc \
#    g++ \
#    python3-dev \
#    libopenblas-dev \
#    liblapack-dev \
#    && rm -rf /var/lib/apt/lists/* """

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
#RUN pip install --no-cache-dir torch==2.3.1
#RUN pip install --no-cache-dir torch-geometric==2.5.3
RUN pip install --no-cache-dir -r requirements.txt

# Scripts and models directories are volume-mounted at runtime
# so we don't COPY them here — changes in ./scripts are live immediately

CMD ["tail", "-f", "/dev/null"]
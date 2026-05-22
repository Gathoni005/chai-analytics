FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev

RUN pip install --upgrade pip

# PyG dependencies
RUN pip install \
    pyg-lib \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.1+cpu.html

# PyG
RUN pip install torch-geometric \
    && pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN python -c "import torch_geometric; print('PyG OK')"

CMD ["tail", "-f", "/dev/null"]

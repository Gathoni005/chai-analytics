FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# 1. Torch first — everything else depends on it
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 2. PyG companion wheels — must match torch version exactly
RUN pip install --no-cache-dir \
    pyg-lib \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.1+cpu.html

# 3. PyG itself
RUN pip install --no-cache-dir torch-geometric

# 4. Rest of your dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Application code last — keeps all dep layers cached on code changes
COPY . .

# 6. Verify
RUN python -c "import torch; import torch_geometric; print('Torch:', torch.__version__, '| PyG OK')"

CMD ["tail", "-f", "/dev/null"]
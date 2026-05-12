%%writefile Dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "train_pinn_ch.py"]

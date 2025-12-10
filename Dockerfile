# -------- Base Python image --------
FROM python:3.10-slim

# -------- System dependencies --------
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -------- Set work directory --------
WORKDIR /app

# -------- Copy requirements --------
COPY requirements.txt .

# -------- Install Python dependencies --------
RUN pip install --no-cache-dir -r requirements.txt

# -------- Copy app code --------
COPY . .

# -------- Expose port --------
EXPOSE 8080

# -------- Run Streamlit --------
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]

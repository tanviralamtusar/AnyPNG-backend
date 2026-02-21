# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required by OpenCV and EasyOCR
# FIXED: Changed libgl1-mesa-glx to libgl1 for newer Debian versions
RUN apt-get update && apt-get install -y \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Download the EDSR 2x Upscaling Model directly into the image
# Download the EDSR 2x and 4x Upscaling Models
RUN wget https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb \
    && wget https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY main.py .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the API (Using shell format)
CMD uvicorn main:app --host 0.0.0.0 --port 8000


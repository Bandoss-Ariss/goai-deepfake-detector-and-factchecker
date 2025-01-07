# Use the official Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for OpenCV and other required tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

ENV TWILIO_SSID=
ENV TWILIO_SECRET_KEY=

EXPOSE 8080


# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

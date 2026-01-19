# Use a newer 3.8 image (Bullseye) to avoid the 404 repository errors
FROM python:3.8-slim-bullseye

# Set the working directory
WORKDIR /app

# Install system dependencies
# These are needed for image processing (PIL/OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app files
COPY . .

# Hugging Face default port
EXPOSE 7860

# Run the app - ensure app.py has host='0.0.0.0' and port=7860
CMD ["python", "app.py"]
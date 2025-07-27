# app/Dockerfile
# This Dockerfile builds a Docker image for the Building Health Monitor Flask application.
# It includes the Python environment, application code, and necessary dependencies.

# Use a lightweight official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the Python dependency file into the container
# This is done first to leverage Docker's build cache. If requirements.txt doesn't change,
# this layer won't be rebuilt, speeding up subsequent builds.
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- IMPORTANT CHANGE: Copy all application code first ---
# This ensures model_trainer.py and app.py are available BEFORE we try to run them.
COPY . .

# --- Now, train ML model during image build ---
# This ensures the model.pkl is available when the app starts.
RUN python model_trainer.py

# Expose port 5000, which the Flask application will listen on
EXPOSE 5000

# Command to run the Flask application when the container starts
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
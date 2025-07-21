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
# --no-cache-dir: Prevents pip from storing downloaded packages, reducing image size.
# -r: Install from the specified requirements file.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# This copies app.py, data_simulator.py, model_trainer.py, etc.
COPY . .

# Expose port 5000, which the Flask application will listen on
EXPOSE 5000

# Command to run the Flask application when the container starts
# Gunicorn is a production-ready WSGI HTTP server.
# -b 0.0.0.0:5000: Binds the server to all network interfaces on port 5000.
# app:app: Specifies the Flask application instance (app object in app.py).
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]

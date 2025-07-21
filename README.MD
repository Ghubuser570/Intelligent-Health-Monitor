Building Health Monitor: Real-time Anomaly Detection System
This project demonstrates an end-to-end MLOps pipeline for real-time anomaly detection, designed to run entirely offline (with GitHub as the only online dependency for initial setup). It simulates monitoring "building health" metrics and flags unusual patterns using a machine learning model, all orchestrated by Docker, automated by Jenkins, and monitored by Prometheus and Grafana.

Project Overview
The system consists of the following components:

app/: A Python Flask application that simulates sensor data ingestion, performs real-time anomaly detection using a pre-trained ML model, and exposes metrics for Prometheus. It also serves a simple web UI to display detected anomalies.

data_simulator.py: A Python script within the app/ directory that generates simulated sensor data and sends it to the Flask application.

model_trainer.py: A Python script within the app/ directory responsible for training (or retraining) the anomaly detection ML model.

Jenkins: An automation server that builds Docker images, retrains the ML model, and deploys the updated application automatically whenever code changes are detected (simulated for local Git changes).

Prometheus: A monitoring system that collects metrics from the app and other services.

Grafana: A visualization tool that creates interactive dashboards from the metrics collected by Prometheus, allowing real-time monitoring of anomalies and system health.

Docker & Docker Compose: Used to containerize all services, ensuring a consistent and "no-hassle" installation and deployment experience.

Getting Started
Follow these steps to set up and run the entire project on your local machine.

Prerequisites
Before you begin, ensure you have the following installed on your system:

Git: For cloning the project repository.

Download: https://git-scm.com/downloads

Docker Desktop: Includes Docker Engine and Docker Compose, essential for running all project services.

Download: https://www.docker.com/products/docker-desktop/

Ensure Docker Desktop is running before proceeding.

Setup Instructions
Clone the Repository:
Open your terminal or command prompt and clone this project to your local machine:

git clone <YOUR_GITHUB_REPO_URL> # Replace with your actual GitHub repo URL
cd building-health-monitor

(Note: For this exercise, you'll copy the provided files into a new local folder and treat it as if you cloned it.)

Build and Start Services with Docker Compose:
Navigate to the root directory of the cloned project (building-health-monitor/) in your terminal and run:

docker compose build
docker compose up -d

docker compose build: This command builds the Docker images for your application and Jenkins based on their respective Dockerfiles. This might take a few minutes the first time.

docker compose up -d: This command starts all the services defined in docker-compose.yml in detached mode (in the background).

Initial ML Model Training:
The Flask app needs a pre-trained model to start. We'll manually trigger the training script once.
First, you need to find the container ID of your app service:

docker ps | grep building-health-monitor-app-1

Look for the CONTAINER ID. Then, execute the training script inside the running app container:

docker exec -it <CONTAINER_ID_OF_APP> python model_trainer.py

This will train the initial Isolation Forest model and save it.

Access the Applications:
Once all services are up and running (this might take a few minutes for Jenkins to fully initialize), you can access them via your web browser:

Building Health Monitor App (Flask): http://localhost:5000

This will show the real-time data and anomaly detections.

Jenkins: http://localhost:8080

You'll need to unlock Jenkins the first time. Follow the instructions on the Jenkins page to get the initial admin password from the container logs:

docker logs jenkins_master

Look for a long alphanumeric string. Create an admin user when prompted.

Prometheus: http://localhost:9090

You can explore the metrics collected here.

Grafana: http://localhost:3000

Default Credentials: admin/admin (you'll be prompted to change this on first login).

The Building Health Dashboard should be automatically provisioned.

Start Data Simulation:
To see data flowing and anomalies being detected, you need to run the data simulator.
Find the container ID of your app service again (if you don't remember it):

docker ps | grep building-health-monitor-app-1

Then, execute the data simulator script inside the running app container:

docker exec -it <CONTAINER_ID_OF_APP> python data_simulator.py

This script will continuously send simulated data to your Flask app. Keep this terminal window open.

Using the Jenkins CI/CD Pipeline
The Jenkins setup includes a pre-configured job (building-health-monitor-pipeline). This job is set up to monitor changes in your local app/ directory (simulating a Git push).

Access Jenkins: Go to http://localhost:8080 and log in.

Navigate to the Job: On the left sidebar, click on Open Blue Ocean or Dashboard. Find the building-health-monitor-pipeline job.

Triggering the Pipeline:

Manual Trigger: You can manually trigger a build by clicking "Build Now" or "Run" within the job's page.

Automated Trigger (Simulated Git Push): To simulate a Git push, make a small change to any file inside the app/ directory (e.g., add a comment to app.py), save it. Jenkins is configured to poll for changes every minute, so it should automatically detect the change and trigger a new pipeline run.

Observe: Watch the Jenkins pipeline execute. It will build a new Docker image for your app, retrain the model, and redeploy the app.

Stopping the Project
To stop all services and remove the Docker containers:

docker compose down

To remove all volumes (which store persistent data like Jenkins configuration and Grafana data):

docker compose down -v

(Use -v if you want a clean slate for Jenkins/Grafana data, but be aware it will reset their configurations and logins.)
Building Health Monitor: Real-time Anomaly Detection System
This project demonstrates an end-to-end MLOps pipeline for real-time anomaly detection, designed to run entirely offline (with GitHub as the only online dependency for initial setup). It simulates monitoring "building health" metrics and flags unusual patterns using a machine learning model, all orchestrated by Docker, automated by Jenkins, and monitored by Prometheus and Grafana.

Project Overview
The system consists of the following components:

app/: A Python Flask application that simulates sensor data ingestion, performs real-time anomaly detection using a pre-trained ML model, and exposes metrics for Prometheus. It also serves a simple web UI to display detected anomalies.

data_simulator.py: A Python script within the app/ directory that generates simulated sensor data and sends it to the Flask application.

model_trainer.py: A Python script within the app/ directory responsible for training (or retraining) the anomaly detection ML model.

Jenkins: An automation server that builds Docker images, retrains the ML model, and deploys the updated application automatically whenever code changes are detected (simulated for local Git changes).

Prometheus: A monitoring system that collects metrics from the app and other services.

Grafana: A visualization tool that creates interactive dashboards from the metrics collected by Prometheus, allowing real-time monitoring of anomalies and system health.

Docker & Docker Compose: Used to containerize all services, ensuring a consistent and "no-hassle" installation and deployment experience.

Getting Started
Follow these steps to set up and run the entire project on your local machine.

Prerequisites
Before you begin, ensure you have the following installed on your system:

Git: For cloning the project repository.

Download: https://git-scm.com/downloads

Docker Desktop: Includes Docker Engine and Docker Compose, essential for running all project services.

Download: https://www.docker.com/products/docker-desktop/

Ensure Docker Desktop is running before proceeding.

Setup Instructions
Clone the Repository:
Open your terminal or command prompt and clone this project to your local machine:

git clone <YOUR_GITHUB_REPO_URL> # Replace with your actual GitHub repo URL
cd building-health-monitor

(Note: For this exercise, you'll copy the provided files into a new local folder and treat it as if you cloned it.)

Build and Start Services with Docker Compose:
Navigate to the root directory of the cloned project (building-health-monitor/) in your terminal and run:

docker compose build
docker compose up -d

docker compose build: This command builds the Docker images for your application and Jenkins based on their respective Dockerfiles. This might take a few minutes the first time.

docker compose up -d: This command starts all the services defined in docker-compose.yml in detached mode (in the background).

Initial ML Model Training:
The Flask app needs a pre-trained model to start. We'll manually trigger the training script once.
First, you need to find the container ID of your app service:

docker ps | grep building-health-monitor-app-1

Look for the CONTAINER ID. Then, execute the training script inside the running app container:

docker exec -it <CONTAINER_ID_OF_APP> python model_trainer.py

This will train the initial Isolation Forest model and save it.

Access the Applications:
Once all services are up and running (this might take a few minutes for Jenkins to fully initialize), you can access them via your web browser:

Building Health Monitor App (Flask): http://localhost:5000

This will show the real-time data and anomaly detections.

Jenkins: http://localhost:8080

You'll need to unlock Jenkins the first time. Follow the instructions on the Jenkins page to get the initial admin password from the container logs:

docker logs jenkins_master

Look for a long alphanumeric string. Create an admin user when prompted.

Prometheus: http://localhost:9090

You can explore the metrics collected here.

Grafana: http://localhost:3000

Default Credentials: admin/admin (you'll be prompted to change this on first login).

The Building Health Dashboard should be automatically provisioned.

Start Data Simulation:
To see data flowing and anomalies being detected, you need to run the data simulator.
Find the container ID of your app service again (if you don't remember it):

docker ps | grep building-health-monitor-app-1

Then, execute the data simulator script inside the running app container:

docker exec -it <CONTAINER_ID_OF_APP> python data_simulator.py

This script will continuously send simulated data to your Flask app. Keep this terminal window open.

Using the Jenkins CI/CD Pipeline
The Jenkins setup includes a pre-configured job (building-health-monitor-pipeline). This job is set up to monitor changes in your local app/ directory (simulating a Git push).

Access Jenkins: Go to http://localhost:8080 and log in.

Navigate to the Job: On the left sidebar, click on Open Blue Ocean or Dashboard. Find the building-health-monitor-pipeline job.

Triggering the Pipeline:

Manual Trigger: You can manually trigger a build by clicking "Build Now" or "Run" within the job's page.

Automated Trigger (Simulated Git Push): To simulate a Git push, make a small change to any file inside the app/ directory (e.g., add a comment to app.py), save it. Jenkins is configured to poll for changes every minute, so it should automatically detect the change and trigger a new pipeline run.

Observe: Watch the Jenkins pipeline execute. It will build a new Docker image for your app, retrain the model, and redeploy the app.

Stopping the Project
To stop all services and remove the Docker containers:

docker compose down

To remove all volumes (which store persistent data like Jenkins configuration and Grafana data):

docker compose down -v

(Use -v if you want a clean slate for Jenkins/Grafana data, but be aware it will reset their configurations and logins.)
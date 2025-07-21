// jenkins/jobs/building-health-monitor-pipeline.groovy
// This Groovy script defines the Jenkins CI/CD pipeline for the Building Health Monitor application.
// It orchestrates building the Docker image, retraining the ML model, and redeploying the app.

// Define the agent where the pipeline will run.
// 'any' means Jenkins can run this on any available agent (including the master itself).
// For this local setup, it will run on the Jenkins master container.
pipeline {
    agent any

    // Define environment variables for the pipeline
    environment {
        // Path to the application directory relative to the Jenkins workspace
        APP_DIR = 'app'
        // Name of the Docker image for the application
        APP_IMAGE_NAME = 'building-health-monitor-app'
        // Name of the Docker container for the application (used for stopping/removing old container)
        APP_CONTAINER_NAME = 'building-health-monitor-app-1'
    }

    // Define stages of the CI/CD pipeline
    stages {
        // --- Stage 1: Checkout Code ---
        // Clones the source code from the Git repository.
        // For this local setup, we're assuming the entire project directory is mounted
        // into the Jenkins container, so we just use the local path.
        stage('Checkout Code') {
            steps {
                script {
                    echo 'Checking out code from local directory...'
                    // In a real scenario, you would clone from GitHub:
                    // git branch: 'main', credentialsId: 'your-github-credentials', url: 'https://github.com/your-user/your-repo.git'
                    // For this setup, the code is already available in the mounted volume.
                    // We just ensure the workspace is clean.
                    sh 'ls -la' // List contents to confirm directory is accessible
                }
            }
        }

        // --- Stage 2: Build Docker Image ---
        // Builds a new Docker image for the Python application.
        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building Docker image: ${APP_IMAGE_NAME}..."
                    // Execute Docker build command.
                    // -t: Tag the image with a name.
                    // ${APP_DIR}: The build context (path to the Dockerfile and application code).
                    sh "docker build -t ${APP_IMAGE_NAME} ${APP_DIR}"
                }
            }
        }

        // --- Stage 3: Train/Retrain ML Model ---
        // Executes the model_trainer.py script inside the newly built Docker image.
        // This ensures the model is always trained with the latest code/dependencies.
        stage('Train ML Model') {
            steps {
                script {
                    echo 'Training/Retraining ML model...'
                    // Run the model_trainer.py script using the new Docker image.
                    // --rm: Automatically remove the container when it exits.
                    // -v: Mount the app directory as a volume to ensure model.pkl is saved to the host.
                    sh "docker run --rm -v ${PWD}/${APP_DIR}:/app ${APP_IMAGE_NAME} python /app/model_trainer.py"
                }
            }
        }

        // --- Stage 4: Deploy Application ---
        // Stops the old application container and starts a new one with the updated image.
        stage('Deploy Application') {
            steps {
                script {
                    echo 'Deploying updated application...'
                    // Stop and remove the old container if it's running
                    sh "docker stop ${APP_CONTAINER_NAME} || true" // '|| true' prevents pipeline failure if container not found
                    sh "docker rm ${APP_CONTAINER_NAME} || true"

                    // Start a new container from the updated image
                    // -d: Run in detached mode (in the background)
                    // -p: Port mapping from host to container
                    // --name: Assign a specific name to the container
                    // -v: Mount the app directory to persist model.pkl and allow data_simulator.py to run later
                    sh "docker run -d -p 5000:5000 --name ${APP_CONTAINER_NAME} -v ${PWD}/${APP_DIR}:/app ${APP_IMAGE_NAME} gunicorn -b 0.0.0.0:5000 app:app"
                    echo 'Application deployed successfully!'
                }
            }
        }
    }

    // --- Post-build Actions ---
    // Actions to perform after the pipeline finishes, regardless of success or failure.
    post {
        always {
            echo 'Pipeline finished.'
        }
        success {
            echo 'Pipeline executed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs for details.'
        }
    }
}

// jenkins/jobs/building-health-monitor-pipeline.groovy
pipeline {
    agent any
    environment {
        APP_DIR = 'app'
        APP_IMAGE_NAME = 'building-health-monitor-app'
        APP_CONTAINER_NAME = 'building-health-monitor-app-1'
    }
    stages {
        stage('Checkout Code') {
            steps {
                script {
                    echo 'Checking out code from local directory...'
                    sh 'ls -la'
                }
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building Docker image: ${APP_IMAGE_NAME}..."
                    sh "docker build -t ${APP_IMAGE_NAME} ${APP_DIR}"
                }
            }
        }
        stage('Train ML Model') {
            steps {
                script {
                    echo 'Training/Retraining ML model...'
                    sh "docker run --rm -v ${PWD}/${APP_DIR}:/app ${APP_IMAGE_NAME} python /app/model_trainer.py"
                }
            }
        }
        stage('Deploy Application') {
            steps {
                script {
                    echo 'Deploying updated application...'
                    sh "docker stop ${APP_CONTAINER_NAME} || true"
                    sh "docker rm ${APP_CONTAINER_NAME} || true"
                    sh "docker run -d -p 5000:5000 --name ${APP_CONTAINER_NAME} -v ${PWD}/${APP_DIR}:/app ${APP_IMAGE_NAME} gunicorn -b 0.0.0.0:5000 app:app"
                    echo 'Application deployed successfully!'
                }
            }
        }
    }
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

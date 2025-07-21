# jenkins/Dockerfile
# This Dockerfile builds a custom Jenkins image with Docker client installed.
# This allows Jenkins to execute Docker commands (e.g., build and run containers)
# from within its own container, which is essential for our CI/CD pipeline.

# Use the official Jenkins LTS (Long Term Support) image as the base
FROM jenkins/jenkins:lts-jdk11

# Switch to root user to install Docker and other tools
USER root

# Install Docker CLI and other necessary tools
# apt-get update: Updates the package list
# apt-get install -y: Installs packages without asking for confirmation
#   - apt-transport-https, ca-certificates, curl, gnupg, lsb-release: Required for Docker's official GPG key
#   - docker-ce-cli: The Docker command-line client
#   - git: Needed by Jenkins to clone repositories
# rm -rf /var/lib/apt/lists/*: Cleans up apt cache to reduce image size
RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release git && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

# Switch back to the jenkins user
USER jenkins

# Copy Jenkins plugins list
# This file lists plugins that Jenkins will automatically install on startup.
COPY plugins.txt /usr/share/jenkins/ref/plugins.txt

# Copy Jenkins Configuration as Code (JCasC) file
# This file automatically configures Jenkins jobs and settings on startup.
COPY jenkins.yaml /usr/share/jenkins/ref/jenkins.yaml

# Expose Jenkins default ports (8080 for HTTP, 50000 for agent communication)
EXPOSE 8080
EXPOSE 50000

# The default entrypoint for Jenkins will handle starting the service.
# No CMD instruction is needed here as the base image provides it.

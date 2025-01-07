# Use Miniconda3 as the base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /home/app

# Update apt-get and install necessary packages
RUN apt-get update && apt-get install -y unzip curl

# Install Deta CLI
RUN curl -fsSL https://get.deta.dev/cli.sh | sh

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install

# Install Python dependencies from the requirements.txt file
COPY requirements.txt /dependencies/requirements.txt
RUN pip install --no-cache-dir -r /dependencies/requirements.txt

# Download and install SpaCy model
RUN python -m spacy download en_core_web_sm

# Set environment variables (AWS and custom environment variables)
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV BACKEND_STORE_URI=$BACKEND_STORE_URI
ENV ARTIFACT_ROOT=$ARTIFACT_ROOT

# Command to run the MLflow server
CMD mlflow server -p $PORT \
    --host 0.0.0.0 \
    --backend-store-uri $BACKEND_STORE_URI \
    --default-artifact-root $ARTIFACT_ROOT

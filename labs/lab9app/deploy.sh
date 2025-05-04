#!/bin/bash

# Set variables
PROJECT_ID="reddit-classifier-ml"
IMAGE_NAME="reddit-classifier"
TAG="v1"
ZONE="us-central1-a"
CLUSTER_NAME="reddit-cluster"

# Confirm project and auth
echo "Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID
gcloud auth application-default set-quota-project $PROJECT_ID

# Enable necessary services
echo "Enabling GCP services..."
gcloud services enable container.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Authenticate Docker for GCP
echo "Configuring Docker to push to gcr.io..."
gcloud auth configure-docker

# Build Docker image
echo "Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME:$TAG .

# Push Docker image to GCP Container Registry
echo "Pushing Docker image to gcr.io..."
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME:$TAG

# Create GKE cluster
echo "Creating GKE cluster..."
gcloud container clusters create $CLUSTER_NAME \
  --num-nodes=1 \
  --zone=$ZONE

# Connect kubectl to the new cluster
echo "Getting cluster credentials..."
gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE

# Deploy Kubernetes resources
echo "Applying Kubernetes deployment and service..."
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Wait for external IP
echo "Waiting for external IP..."
sleep 10
kubectl get service reddit-classifier-service

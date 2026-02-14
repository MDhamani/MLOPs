# Docker Labs - MLOps

A collection of MLOps Docker labs focusing on containerization, model training, and deployment best practices.

## Project Overview

This repository contains hands-on labs for learning Docker and MLOps. Currently includes Lab1, which implements a Breast Cancer Detection model using XGBoost with Docker containerization.

### Lab1: Breast Cancer Detection with XGBoost

A machine learning project that trains an XGBoost classifier for breast cancer detection, containerized using Docker with multi-stage builds.

**Key Features:**
- XGBoost classifier for binary classification
- Breast Cancer dataset
- Comprehensive test suite (pytest)
- Multi-stage Docker build for optimized image size
- Python 3.10+ with uv package manager
- Built-in logging

## Prerequisites

### For Local Development
- Python 3.12 or higher
- [uv package manager](https://docs.astral.sh/uv/) (recommended) or pip
- Git

### For Docker
- Docker Engine (any recent version)
- Docker Compose (optional, for orchestration)

## Getting Started

### Local Setup

1. **Clone or navigate to the project**
   ```bash
   cd Lab1
   ```

2. **Install dependencies using uv**
   ```bash
   uv sync
   ```
   
   Or using pip:
   ```bash
   pip install -r src/requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import xgboost; import sklearn"
   ```

### Running Locally

1. **Run the main script**
   ```bash
   python src/main.py
   ```

2. **Run tests**
   ```bash
   pytest src/test.py -v
   ```

3. **Run specific test**
   ```bash
   pytest src/test.py::test_name -v
   ```

## Docker Setup

### Building the Docker Image

1. **Build the image**
   ```bash
   cd Lab1
   docker build -t lab1:v1 .
   ```

2. **Verify the build (optional)**
   ```bash
   docker images | grep lab1
   ```

### Running with Docker

1. **Run the container**
   ```bash
   docker run lab1:v1
   ```

2. **Run with interactive terminal**
   ```bash
   docker run -it lab1:v1 /bin/bash
   ```

3. **Run and mount local volume**
   ```bash
   docker run -v $(pwd):/data lab1:v1
   ```

### Exporting Docker Image

Save the image to a tar file:
```bash
docker save lab1:v1 > lab1_image.tar
```

Load from tar file:
```bash
docker load < lab1_image.tar
```

## Dependencies

- **scikit-learn** - ML utilities and Breast Cancer dataset
- **xgboost** - Gradient boosting classifier
- **joblib** - Model serialization
- **pytest** - Testing framework

## Docker Build Details

The Dockerfile uses a two-stage build:

1. **Builder stage**: Compiles and tests everything
2. **Runtime stage**: Contains only necessary runtime components

This results in smaller, more efficient container images.

## Changes Made by Manav Dhamani for Docker Lab 1

- Changed the model from Random Forest Classifier to XGBoost
- Changed dataset to Breast Cancer Detection
- Added test cases for new dataset
- Switched from pip to uv package manager
- Updated Dockerfile to multi-stage build
- Added logging

## Troubleshooting

**Python version mismatch:**
```bash
python --version  # Should be 3.10+
```

**Docker build fails:**
- Ensure Docker daemon is running
- Check internet connection for dependency downloads
- Clear Docker cache: `docker build --no-cache -t lab1:v1 .`

**Tests fail locally:**
- Verify all dependencies installed: `uv sync` or `pip install -r src/requirements.txt`
- Check Python version compatibility
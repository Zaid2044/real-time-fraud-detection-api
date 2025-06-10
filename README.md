# Project Aether: Real-Time Fraud Detection API

*A high-performance, containerized REST API for real-time credit card fraud detection, built with FastAPI and Scikit-learn, and deployed with Docker.*

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Tech Stack](#tech-stack)
5. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation & Setup](#installation--setup)
6. [API Usage](#api-usage)
   - [Endpoints](#endpoints)
   - [Example Request](#example-request)
7. [Project Structure](#project-structure)
8. [Future Improvements](#future-improvements)

---

## Project Overview

The financial industry loses billions of dollars to credit card fraud annually. The challenge is to detect fraudulent transactions in real-time without impacting the user experience for legitimate transactions. Project Aether addresses this by providing a robust, low-latency API that leverages a machine learning model to score transactions for their likelihood of being fraudulent.

This project demonstrates a T-shaped skill set: deep expertise in building and serving an ML model, combined with a broad, system-level understanding of MLOps, containerization, and API design.

## Key Features

- **Real-Time Inference:** Sub-second response times for fraud prediction.
- **Containerized Deployment:** Packaged with Docker for portability, scalability, and environment consistency.
- **Industry-Standard API:** Built with FastAPI, providing automatic OpenAPI (Swagger UI) and ReDoc documentation.
- **Clean Architecture:** Separation of concerns between the API logic (`src`), model training (`notebooks`), and the final model artifact (`model.pkl`).
- **Data Validation:** Pydantic models ensure that incoming data conforms to the required schema.

## System Architecture

The system is designed for simplicity and scalability. A client sends a POST request with transaction data to the API, which is running inside a Docker container. The FastAPI application processes the request, uses the pre-trained Scikit-learn model to make a prediction, and returns a JSON response.

```text
+----------------+      HTTP POST Request      +---------------------------------+
|                |   (Transaction JSON data)   |      Docker Container           |
|  Client (e.g., | --------------------------->|  +---------------------------+  |
|  Postman/App)  |                             |  |  FastAPI Application      |  |
|                |                             |  | +-----------------------+ |  |
|                |<--------------------------- |  | |       Uvicorn         | |  |
+----------------+   JSON Response (0 or 1)    |  | +-----------------------+ |  |
                      (Fraud/Not Fraud)         |  | |  Prediction Endpoint  | |  |
                                                |  | +-----------------------+ |  |
                                                |  | |   Scikit-learn Model  | |  |
                                                |  | +-----------------------+ |  |
                                                |  +---------------------------+  |
                                                +---------------------------------+


Tech Stack
Backend: FastAPI
ML Library: Scikit-learn, Pandas
Server: Uvicorn
Containerization: Docker
Data Validation: Pydantic
Language: Python 3.10
Getting Started
Follow these instructions to get the API up and running on your local machine.
Prerequisites
Docker Desktop installed and running.
Git LFS for handling the large dataset file.
A tool for making API requests, such as Postman or curl.
Installation & Setup

Clone the repository:
git clone https://github.com/Zaid2044/real-time-fraud-detection-api.git
cd real-time-fraud-detection-api


Build the Docker image:
docker build -t aether-fraud-api .

Run the Docker container:
docker run -d -p 8000:8000 --name aether-api-container aether-fraud-api

The API is now running and accessible at http://localhost:8000.
API Usage

Endpoints
GET /: Root endpoint. Returns a welcome message.
GET /docs: Access the interactive Swagger UI documentation.
POST /predict: The main prediction endpoint. Accepts transaction data and returns a fraud prediction.

Example Request
You can use curl or the /docs page to test the prediction endpoint.
Request Body (POST /predict):
The model expects a JSON object with 30 numerical features (V1-V28, Time, Amount).
{
  "features": [
    -1.359807, -0.07278, 2.53634, 1.37815, -0.33832, 0.46238, 0.23959, 0.09869, 0.36378, 0.09079, -0.55159, -0.61780, -0.87778, 0.40399, -0.40719, 0.91528, -0.08330, -0.99138, -0.31116, 1.46817, -0.47040, 0.20797, 0.02579, 0.40399, 0.16705, -0.04249, 0.28284, -0.00512, 0, 149.62
  ]
}


Example curl command:
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{ "features": [ -1.35, -0.07, 2.53, 1.37, -0.33, 0.46, 0.23, 0.09, 0.36, 0.09, -0.55, -0.61, -0.87, 0.40, -0.40, 0.91, -0.08, -0.99, -0.31, 1.46, -0.47, 0.20, 0.02, 0.40, 0.16, -0.04, 0.28, -0.005, 0, 149.62 ] }'


Success Response (200 OK):
{
  "prediction": 0,
  "is_fraud": "Not Fraud"
}


Project Structure
.
├── .dockerignore         # Files to ignore in the Docker build context
├── .gitattributes        # Configures Git LFS file tracking
├── .gitignore            # Files to ignore by Git
├── data/
│   └── creditcard.csv    # Raw dataset (tracked by Git LFS)
├── Dockerfile            # Instructions to build the Docker image
├── LICENSE
├── model.pkl             # Serialized trained model
├── notebooks/
│   └── 01_initial...ipynb # EDA and model training notebook
├── README.md
├── requirements.txt      # Python dependencies
└── src/
    └── main.py           # FastAPI application source code

Future Improvements
Integrate with a CI/CD pipeline (e.g., GitHub Actions) for automated testing and deployment.
Add a monitoring solution (e.g., Prometheus/Grafana) to track API performance and model drift.
Implement a proper logging framework.
Create a more complex model with feature engineering and hyperparameter tuning.
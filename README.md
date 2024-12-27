# Demand Forecasting API 🚀

This project demonstrates my expertise in building and deploying a scalable Demand Forecasting API using Flask, Docker, and Google Cloud Run. It showcases my ability to create solutions that integrate machine learning with modern cloud infrastructure for practical business use cases.

---

## 🔍 Project Overview

The **Demand Forecasting API** is designed to predict future demand trends based on historical data. This project highlights my skills in:
- **API Development:** Implementing Flask-based APIs for forecasting and data processing.
- **Containerization:** Using Docker to ensure portability and scalability.
- **Cloud Deployment:** Deploying serverless applications on Google Cloud Run.

---

## ✨ Key Features

- **Scalable Cloud Deployment:** Deployed on **Google Cloud Run**, leveraging serverless infrastructure for high availability and automatic scaling.
- **Dockerized Application:** Fully containerized for portability across environments.
- **Batch Forecasting:** Designed to process batch data for demand predictions.
- **Lightweight & Optimized:** Built with Flask for rapid API responses and ease of development.

---

## 🛠️ Technologies Used

- **Flask**: Web framework for building RESTful APIs.
- **Docker**: For containerizing the application.
- **Google Cloud Run**: Serverless cloud platform for deploying and running the API.
- **Python**: Core programming language used for implementation.

---

## ⚙️ How to Set It Up

### Local Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/TishaRepo/Demand-Forecasting-API.git
    cd Demand-Forecasting-API
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    python app.py
    ```

4. Access the API locally at `http://127.0.0.1:5000`.

---

### Deployment on Google Cloud Run
1. Build the Docker image:
    ```bash
    docker build -t gcr.io/<your-project-id>/demand-forecasting-api:1 .
    ```

2. Push the image to Google Container Registry:
    ```bash
    docker push gcr.io/<your-project-id>/demand-forecasting-api:1
    ```

3. Deploy the container to Google Cloud Run:
    ```bash
    gcloud run deploy demand-forecasting-api \
        --image gcr.io/<your-project-id>/demand-forecasting-api:1 \
        --platform managed \
        --region asia-northeast2 \
        --allow-unauthenticated

    ```

4. Use the provided URL to access the API.
The web app can be accessed at "https://demand-forecasting-api-201587345242.asia-northeast2.run.app/"
---

## 🔗 API Endpoints

### `/predict` Endpoint
- **Method:** `POST`
- **Description:** Accepts input data and returns demand forecasts.

**Request Body Example:**
```json
{
  "data": [
    {"feature1": value1, "feature2": value2},
    ...
  ]
}

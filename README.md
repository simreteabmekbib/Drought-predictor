# **Drought Predictor Project**

![Drought](https://github.com/simreteabmekbib/Drought-predictor/blob/main/image.jpg)

An end-to-end machine learning project designed to predict drought levels in the continental US using meteorological and soil data. This project incorporates best practices, including a training pipeline, experiment tracking, deployment, and monitoring.

---

## **Project Description**

This project predicts drought levels using meteorological and soil data collected across the continental United States. The dataset, sourced from Kaggle, provides daily measurements and drought severity classifications.

### **Drought Severity Levels**
- **None**: No drought.  
- **1**: Abnormally dry.  
- **2**: Moderate drought.  
- **3**: Severe drought.  
- **4**: Extreme drought.  
- **5**: Exceptional drought.  

The goal is to classify drought severity using meteorological data, enabling predictions of potential drought conditions.

---

## **Key Features**
- **Data Exploration and Preprocessing**:  
  Initial exploration and preprocessing were performed using Jupyter notebooks.

- **Machine Learning Model**:  
  A Gradient Boosting model was trained for drought classification.

- **Workflow Orchestration**:  
  Used **Prefect** to manage workflows, including data processing and model training.

- **Experiment Tracking**:  
  Managed experiments and model versions using **MLflow**.

- **Containerization**:  
  The model is containerized with **Docker**, allowing flexible deployment. Predictions are served via a Flask application running on a Gunicorn server.

- **Monitoring**:  
  Metrics are tracked with **Prometheus** and visualized using **Grafana**.

- **Code Quality**:  
  Includes unit tests, integration tests, and automated code formatting with `flake8` and `black`.

- **Automation**:  
  Implements pre-commit hooks and a CI/CD pipeline to ensure code quality and automated deployment.

---

## **How to Run the Project**

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/drought-predictor-project.git
    cd drought-predictor-project
    ```

2. Create a `data` directory:
    ```bash
    mkdir data
    ```

3. Download the dataset from Kaggle:  
   [US Drought Meteorological Data](https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data)  
   Place the dataset in the `data` folder.

4. Run the pipeline:
    ```bash
    make all
    ```

5. Test the deployed model:
    ```bash
    python scripts/send_prediction_request.py
    ```

---

## **Access Project Interfaces**

- **MLflow** (Experiment Tracking): [http://localhost:5001/](http://localhost:5001/)  
- **Prefect** (Workflow Orchestration): [http://localhost:4200/](http://localhost:4200/)  
- **Prometheus** (Metrics Monitoring): [http://localhost:9090/](http://localhost:9090/)  
- **Grafana** (Metrics Visualization): [http://localhost:3000/](http://localhost:3000/)  
- **Deployed Drought Predictor**: [http://localhost:5002/](http://localhost:5002/)  

---

## **Tools and Technologies Used**

| **Tool**       | **Purpose**                         | **Link**                                      |
|-----------------|-------------------------------------|----------------------------------------------|
| **MLflow**      | Experiment Tracking                | [MLflow Docs](https://mlflow.org/)           |
| **Prefect**     | Workflow Orchestration             | [Prefect Docs](https://www.prefect.io/)      |
| **Docker**      | Containerization                   | [Docker Docs](https://www.docker.com/)       |
| **Prometheus**  | Monitoring                         | [Prometheus Docs](https://prometheus.io/)    |
| **Grafana**     | Metrics Visualization              | [Grafana Docs](https://grafana.com/)         |
| **Flask**       | Web Application Framework          | [Flask Docs](https://flask.palletsprojects.com/) |

---

## **Steps Followed**

### Initial Setup
1. Install dependencies:
    ```bash
    pipenv install -r requirements.txt
    pipenv shell
    ```

2. Organize project structure:
    - Create folders: `data`, `models`, `scripts`.
    - Add files: `Dockerfile`, `docker-compose.yml`, `Makefile`.

3. Data exploration and preprocessing in Jupyter Notebook.

### Model Training and Tracking
4. Train the model in Jupyter Notebook and track experiments in MLflow:
    ```bash
    pipenv run mlflow ui
    ```

5. Export training code to `train.py` and define key functions (load, clean, train, save).

6. Use Prefect for workflow orchestration:
    ```bash
    prefect server start
    python scripts/train.py
    ```

### Deployment
7. Convert `predict.py` into a Flask application for containerized deployment.

8. Build and run the Docker container:
    ```bash
    docker build -t drought_predictor .
    docker run -d -p 5002:5002 --name drought_predictor drought_predictor
    ```

9. Test the deployed model with:
    ```bash
    python scripts/send_prediction_request.py
    ```

### Monitoring and Visualization
10. Record metrics with Prometheus and visualize them in Grafana:
    - Prometheus: [http://localhost:9090/](http://localhost:9090/)  
    - Grafana: [http://localhost:3000/](http://localhost:3000/)  
    - Model Metrics: [http://localhost:5002/metrics](http://localhost:5002/metrics)

### Quality Assurance
11. Add unit tests and integration tests.

12. Apply linter and code formatter:
    ```bash
    flake8
    black .
    ```

13. Automate pre-commit checks:
    ```bash
    pre-commit install
    ```

14. Implement CI/CD pipeline with GitHub Actions (`.github/workflows/ci_cd.yml`).

---
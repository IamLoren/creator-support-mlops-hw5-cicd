# Homework 5 — CI/CD for Machine Learning

CI/CD pipeline for the **customer-support intent classification** project.

The goal of this homework is to automate the delivery flow of an ML model:

```text
dataset
  ↓
training
  ↓
model artifact
  ↓
Docker image
  ↓
inference service
  ↓
deployment artifact
```

The pipeline is implemented with **GitHub Actions** and automatically performs model training and deployment-related steps.

---

## Project overview

The ML task is multiclass intent classification of customer-support messages in Ukrainian and English.

The model uses:

```text
TF-IDF
  ↓
Logistic Regression
```

The dataset is versioned with **DVC** and stored remotely in **Cloudflare R2**.

The final inference service is implemented with **FastAPI** and packaged as a Docker image.

The deployment image is published to **GitHub Container Registry (GHCR)**.

---

# CI/CD architecture

```text
GitHub push / manual trigger
            ↓
      GitHub Actions
            ↓
        TRAIN JOB
            ↓
      Checkout repository
            ↓
    Install dependencies
            ↓
       Configure DVC
            ↓
   Pull dataset from R2
            ↓
      Validate dataset
            ↓
        Train model
            ↓
      Evaluate model
            ↓
   Create model.joblib
            ↓
 Upload GitHub Actions artifact
            ↓
        DEPLOY JOB
            ↓
 Download trained artifact
            ↓
      Build Docker image
            ↓
 Start inference container
            ↓
       Health check
            ↓
 Prediction smoke test
            ↓
       Login to GHCR
            ↓
 Build + push deployment image
            ↓
   GitHub Container Registry
```

---

# Repository structure

```text
creator-support-mlops-hw5-cicd/
│
├── .github/
│   └── workflows/
│       └── ml-cicd.yml
│
├── .dvc/
│   ├── config
│   └── .gitignore
│
├── data/
│   └── annotated/
│       ├── .gitignore
│       └── customer_support_intents.json.dvc
│
├── inference/
│   └── app.py
│
├── scripts/
│   └── validate_dataset.py
│
├── training/
│   └── train.py
│
├── Dockerfile
├── requirements-training.txt
├── requirements-inference.txt
├── .dockerignore
├── .dvcignore
├── .gitignore
└── README.md
```

---

# Dataset

Dataset version:

```text
dataset-v0.2
```

Dataset path:

```text
data/annotated/customer_support_intents.json
```

The dataset contains:

```text
144 messages
72 semantic scenarios
72 Ukrainian messages
72 English messages
9 intent classes
4 business domains
```

The dataset itself is not stored directly in Git.

Git stores only the DVC pointer:

```text
data/annotated/customer_support_intents.json.dvc
```

The actual dataset is stored in Cloudflare R2 and restored with:

```bash
dvc pull
```

---

# Dataset validation

Before training, the CI pipeline validates the dataset:

```bash
python scripts/validate_dataset.py \
  --input data/annotated/customer_support_intents.json \
  --expected-records 144 \
  --expected-scenarios 72
```

The validation checks include:

```text
record count
scenario count
unique IDs
allowed languages
allowed domains
allowed intents
required fields
scenario consistency
```

Expected result:

```text
Dataset validation PASSED
Records: 144
Scenarios: 72
```

If validation fails, the training stage stops.

---

# Training

Training is implemented in:

```text
training/train.py
```

The model pipeline is:

```text
TfidfVectorizer
        ↓
LogisticRegression
```

The selected hyperparameters come from Homework 2:

```text
LogisticRegression C = 10.0
TF-IDF ngram_range = (1, 1)
max_iter = 1000
```

The dataset is split at the **scenario level**.

This is important because Ukrainian and English versions of the same semantic scenario share the same `scenario_id`.

They must never appear on opposite sides of the train/test split.

The split used is:

```text
development scenarios: 54
test scenarios: 18
```

which corresponds to:

```text
development records: 108
test records: 36
```

---

# Training output

The training script creates:

```text
artifacts/
├── model.joblib
├── metrics.json
├── model_metadata.json
└── classification_report.json
```

These files are generated during CI and are not committed to Git.

---

## Model artifact

```text
artifacts/model.joblib
```

contains the complete trained scikit-learn pipeline:

```text
TF-IDF vectorizer
+
Logistic Regression classifier
```

The pipeline can therefore be loaded directly for inference.

---

## Metrics

The final test metrics are approximately:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.75 |
| Macro F1 | 0.7422 |
| Weighted F1 | 0.7422 |

Example `metrics.json`:

```json
{
  "accuracy": 0.75,
  "macro_f1": 0.7421516754850088,
  "weighted_f1": 0.7421516754850088
}
```

---

# Inference service

The inference API is implemented in:

```text
inference/app.py
```

It loads:

```text
artifacts/model.joblib
```

during application startup.

The service exposes two main endpoints.

---

## Health endpoint

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_path": "artifacts/model.joblib",
  "model_loaded": true
}
```

---

## Prediction endpoint

```http
POST /predict
```

Request:

```json
{
  "text": "I cannot access my account."
}
```

Example response:

```json
{
  "intent": "ACCESS_ACCOUNT"
}
```

---

# Docker image

The inference service is packaged using:

```text
Dockerfile
```

The Docker image contains:

```text
FastAPI application
trained model.joblib
model metadata
runtime dependencies
```

The training environment is not included in the final inference image.

This keeps training and serving responsibilities separated.

---

# Local setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows Git Bash:

```bash
source .venv/Scripts/activate
```

Install training dependencies:

```bash
python -m pip install -r requirements-training.txt
```

---

# Restore the dataset locally

The local DVC credentials are stored in:

```text
.dvc/config.local
```

This file contains private R2 access credentials and must never be committed.

Restore the dataset:

```bash
dvc pull
```

Validate it:

```bash
python scripts/validate_dataset.py \
  --input data/annotated/customer_support_intents.json \
  --expected-records 144 \
  --expected-scenarios 72
```

---

# Train locally

Run:

```bash
python training/train.py
```

Expected output includes approximately:

```text
Total records: 144
Development records: 108
Test records: 36

Development scenarios: 54
Test scenarios: 18

Test metrics:
Accuracy:    0.7500
Macro F1:    0.7422
Weighted F1: 0.7422
```

Generated artifacts:

```text
artifacts/model.joblib
artifacts/metrics.json
artifacts/model_metadata.json
artifacts/classification_report.json
```

---

# Run inference locally

Install inference dependencies:

```bash
python -m pip install -r requirements-inference.txt
```

Start the service:

```bash
python -m uvicorn inference.app:app \
  --host 127.0.0.1 \
  --port 8000
```

Test health:

```bash
curl http://127.0.0.1:8000/health
```

Test prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I cannot access my account."}'
```

Expected response:

```json
{
  "intent": "ACCESS_ACCOUNT"
}
```

---

# Build Docker image locally

After training:

```bash
docker build \
  -t customer-support-cicd-api:local \
  .
```

Run:

```bash
docker run --rm \
  --name customer-support-cicd-api \
  -p 8000:8000 \
  customer-support-cicd-api:local
```

Test:

```bash
curl http://127.0.0.1:8000/health
```

and:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I cannot access my account."}'
```

---

# GitHub Actions pipeline

The CI/CD workflow is defined in:

```text
.github/workflows/ml-cicd.yml
```

It contains two main jobs:

```text
train
deploy
```

The `deploy` job depends on `train`:

```text
train
  ↓
deploy
```

If training fails, deployment does not start.

---

# Pipeline triggers

The pipeline supports two triggers.

## Push to main

```text
push → main
```

Any push to the `main` branch automatically starts the pipeline.

---

## Manual trigger

The workflow also supports:

```text
workflow_dispatch
```

This allows the pipeline to be launched manually from:

```text
GitHub
→ Actions
→ ML CI/CD Pipeline
→ Run workflow
```

This trigger is convenient for testing and demonstrations.

---

# Train job

The `train` job performs:

```text
1. Checkout repository
2. Set up Python
3. Install training dependencies
4. Configure DVC remote
5. Pull dataset
6. Validate dataset
7. Train model
8. Print training metrics
9. Verify artifacts
10. Upload trained model as GitHub Actions artifact
```

The uploaded artifact is named:

```text
trained-model
```

and contains:

```text
model.joblib
model_metadata.json
metrics.json
classification_report.json
```

---

# Passing the model between jobs

GitHub Actions jobs run on separate temporary runners.

Therefore, files created in the `train` job do not automatically exist in the `deploy` job.

The pipeline explicitly transfers them using GitHub Actions artifacts:

```text
train runner
     ↓
actions/upload-artifact
     ↓
GitHub artifact storage
     ↓
actions/download-artifact
     ↓
deploy runner
```

This is how the trained model moves from training to deployment.

---

# Deploy job

The `deploy` job performs:

```text
1. Checkout repository
2. Download trained model artifact
3. Verify model files
4. Build inference Docker image
5. Start inference container
6. Wait for /health
7. Perform prediction smoke test
8. Log in to GHCR
9. Build and push final image
```

---

# Health check

After starting the Docker container, GitHub Actions repeatedly checks:

```http
GET /health
```

The deployment continues only after the inference API responds successfully.

If the service does not become ready, the job fails.

---

# Prediction smoke test

The pipeline sends a real prediction request:

```json
{
  "text": "I cannot access my account."
}
```

and verifies that the response contains:

```text
ACCESS_ACCOUNT
```

This confirms that:

```text
Docker container starts
        +
model loads
        +
FastAPI works
        +
model inference works
```

before the deployment image is published.

---

# GitHub Container Registry

The final deployment image is published to:

```text
ghcr.io/iamloren/creator-support-mlops-api
```

Two tags are generated.

Stable/latest tag:

```text
ghcr.io/iamloren/creator-support-mlops-api:latest
```

Immutable commit-specific tag:

```text
ghcr.io/iamloren/creator-support-mlops-api:<git-sha>
```

The SHA tag provides traceability between:

```text
Git commit
   ↓
workflow run
   ↓
Docker image
```

---

# Pull the deployed image

The image created by GitHub Actions can be downloaded independently from the local build:

```bash
docker pull \
  ghcr.io/iamloren/creator-support-mlops-api:latest
```

Run it:

```bash
docker run --rm \
  --name customer-support-ghcr-api \
  -p 8001:8000 \
  ghcr.io/iamloren/creator-support-mlops-api:latest
```

Port `8001` is used here to avoid conflicts if another local inference service already occupies port `8000`.

---

## Test the deployed GHCR image

Health:

```bash
curl http://127.0.0.1:8001/health
```

Prediction:

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I cannot access my account."}'
```

Expected response:

```json
{
  "intent": "ACCESS_ACCOUNT"
}
```

This proves that the image produced and published by GitHub Actions can be pulled and run independently.

---

# GitHub Secrets

The training job needs credentials to access the private Cloudflare R2 DVC storage.

The following repository secrets are configured in:

```text
Settings
→ Secrets and variables
→ Actions
```

Secret names:

```text
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
R2_ENDPOINT_URL
```

The values are never stored in Git.

The workflow temporarily writes them to DVC local configuration on the GitHub runner.

---

# Security

Sensitive files are excluded from Git, including:

```text
.dvc/config.local
.env
.venv/
artifacts/
data/annotated/customer_support_intents.json
```

The repository contains only:

```text
DVC metadata
source code
Docker configuration
GitHub Actions workflow
documentation
```

No R2 access key or secret key is committed.

---

# What "deployment" means in this homework

The pipeline creates a production-ready deployment artifact:

```text
trained ML model
        ↓
Docker inference image
        ↓
GitHub Container Registry
```

The GitHub Actions runner also starts the image and verifies it using health and prediction smoke tests.

The homework does not provision a permanent external cloud server.

The deployment target in this implementation is **GHCR**, from which the tested Docker image can be pulled by a runtime environment.

---

# Reproducibility

A fresh environment can reproduce the complete pipeline from:

```text
Git repository
+
DVC remote dataset
+
GitHub Secrets
```

GitHub Actions then automatically performs:

```text
DVC pull
   ↓
dataset validation
   ↓
training
   ↓
evaluation
   ↓
artifact creation
   ↓
Docker build
   ↓
health test
   ↓
prediction test
   ↓
GHCR publication
```

No local model file is required.

---

# Homework requirements

Required:

- [x] CI/CD pipeline
- [x] GitHub Actions
- [x] automatic trigger
- [x] manual `workflow_dispatch` trigger
- [x] model training job
- [x] deployment job
- [x] training actually runs in CI
- [x] dataset automatically restored through DVC
- [x] dataset validation
- [x] trained model passed between jobs
- [x] Docker inference image
- [x] health check
- [x] real prediction smoke test
- [x] deployment image published to GHCR
- [x] commit-specific image version
- [x] README with pipeline instructions
- [ ] demo video

---

# Demo flow

For the homework video, the complete CI/CD process can be demonstrated as:

```text
1. Open GitHub Actions

2. Run:
   ML CI/CD Pipeline
   using workflow_dispatch

3. Show TRAIN job:
   - DVC pull
   - dataset validation
   - model training
   - metrics
   - model artifact upload

4. Show DEPLOY job:
   - artifact download
   - Docker build
   - container start
   - health check
   - prediction smoke test
   - GHCR push

5. Show both jobs green

6. Open GitHub Packages
   and show:
   creator-support-mlops-api

7. Locally run:
   docker pull ghcr.io/iamloren/creator-support-mlops-api:latest

8. Start the downloaded image

9. Send POST /predict

10. Show:
    {"intent":"ACCESS_ACCOUNT"}
```

This demonstrates the complete automated flow:

```text
training
   ↓
deployment
```

required by Homework 5.
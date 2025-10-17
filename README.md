# Northwestern University - MLDS 400 - HW3 

This repository provides two runnable pipelines for the Kaggle Titanic dataset. One pipeline is in Python. One pipeline is in R. Both print clear step logs and write outputs to the artifacts folder. The repository does not include any data files.

## Repository layout

1. Dockerfile at repository root for Python
2. requirements.txt for Python
3. src/train.py
4. src/download_data.py
5. src/r/Dockerfile for R
6. src/r/install_packages.R
7. src/r/train.R
8. src/data is where you place CSV files, and Git ignores it
9. artifacts is where outputs are written at runtime

## Prerequisites

1. Docker Desktop is running on Windows or macOS. Docker Engine is running on Linux
2. Optional local run uses Python 3.11 or newer
3. Internet access during image build

Windows users should enable the WSL 2 based engine in Docker Desktop settings.

## Dataset location and download

1. Sign in to Kaggle
2. Open the Titanic dataset page
3. Download train.csv and test.csv
4. Place both files in src/data on your local clone

Do not commit any data files. If the CSV files are missing, the Python pipeline can download a single file fallback named titanic.csv by using src/download_data.py.

## Quick start with Docker

All commands run from the repository root. These commands print the step logs and write outputs to artifacts.

Python on PowerShell
```powershell
docker build -t mlds-hw3:latest .
docker run --rm -it -v "${PWD}\src\data:/app/src/data" -v "${PWD}\artifacts:/app/artifacts" mlds-hw3:latest
````

Python on Git Bash or macOS or Linux

```bash
docker build -t mlds-hw3:latest .
docker run --rm -it -v "$(pwd)/src/data:/app/src/data" -v "$(pwd)/artifacts:/app/artifacts" mlds-hw3:latest
```

R on PowerShell

```powershell
docker build -f src/r/Dockerfile -t mlds-hw3-r:latest .
docker run --rm -it -v "${PWD}\src\data:/app/src/data" -v "${PWD}\artifacts:/app/artifacts" mlds-hw3-r:latest
```

R on Git Bash or macOS or Linux

```bash
docker build -f src/r/Dockerfile -t mlds-hw3-r:latest .
docker run --rm -it -v "$(pwd)/src/data:/app/src/data" -v "$(pwd)/artifacts:/app/artifacts" mlds-hw3-r:latest
```

## Example output  Python  abridged

```
[STEP 14] Detected local data files: ['train.csv', 'test.csv']
[STEP 15] Train shape: (891, 8)
[STEP 16] Feature columns used: [...]
[STEP 17] Training accuracy: 0.7991
[STEP 18] Holdout accuracy: 0.8045
[STEP 18] Wrote submission file: artifacts/submission.csv
```

## Example output  R  abridged

```
[STEP 14] Detected data files: train.csv, test.csv
[STEP 15] Train shape: (891, 8)
[STEP 17] Training accuracy: 0.7980
[STEP 18] Holdout accuracy: 0.7598
[STEP 19] Saved R model to: /app/artifacts/model_r_glm.rds
[STEP 20] Wrote submission: /app/artifacts/submission_r.csv
```

Note on evaluation
Kaggle test.csv does not include labels. The scripts report training accuracy and an eighty-twenty (80-20) holdout accuracy split from train.csv. The submission CSV files are intended for Kaggle scoring.

## Outputs

Python outputs

1. artifacts/model.joblib
2. artifacts/submission.csv if test.csv exists

R outputs

1. artifacts/model_r_glm.rds
2. artifacts/submission_r.csv if test.csv exists

## Branching and pull requests

1. Work on a feature branch
2. Open a pull request from the feature branch to develop
3. Open a pull request from develop to main
4. This is equivalent to submitting from develop to master as written in the prompt

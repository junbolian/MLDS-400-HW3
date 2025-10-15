Data placement
- Preferred: put Kaggle files `train.csv`, `test.csv`, `gender_submission.csv` under `src/data/`.
- If absent, the code will auto-download a single-file fallback `titanic.csv` and run a train/test split.
- The repository never commits data files; `src/data/` is git-ignored.

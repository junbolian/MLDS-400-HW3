# src/r/train.R
# Part 4 (R) — Repeat steps 14–21 with clear console logs.
s <- function(n, msg) cat(sprintf("[STEP %d] %s\n", n, msg))

s(14, "R runtime starting...")

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(tibble)
})

root <- normalizePath("/app", mustWork = FALSE)
data_dir <- file.path(root, "src", "data")
art_dir  <- file.path(root, "artifacts")

if (!dir.exists(art_dir)) dir.create(art_dir, recursive = TRUE, showWarnings = FALSE)

train_path <- file.path(data_dir, "train.csv")
test_path  <- file.path(data_dir, "test.csv")

if (!file.exists(train_path)) {
  s(14, sprintf("train.csv not found at %s", train_path))
  stop("No train.csv; please place Kaggle files under src/data/")
} else {
  s(14, sprintf("Detected data files: %s%s",
                basename(train_path),
                if (file.exists(test_path)) paste0(", ", basename(test_path)) else ""))
}

# ---- STEP 15: Load and light EDA / cleaning ----
read_csv_robust <- function(path) {
  # Try comma first (Kaggle default), then semicolon, then tab
  tries <- list(
    function() readr::read_csv(path, show_col_types = FALSE, progress = FALSE),
    function() readr::read_csv2(path, show_col_types = FALSE, progress = FALSE),
    function() readr::read_delim(path, delim = "\t", show_col_types = FALSE, progress = FALSE)
  )
  last_err <- NULL
  for (f in tries) {
    out <- tryCatch(f(), error = function(e) e)
    if (!inherits(out, "error")) return(out)
    last_err <- out
  }
  stop(sprintf("Failed to read %s: %s", path, as.character(last_err)))
}

s(15, "Loading Kaggle train.csv")
df <- read_csv_robust(train_path)

# Normalize column names (strip, lowercase)
norm_names <- function(x) {
  x |>
    str_replace_all("\\ufeff", "") |>
    str_trim() |>
    str_replace_all("[^A-Za-z0-9]+", "_") |>
    tolower()
}
names(df) <- norm_names(names(df))

# Minimal required columns
need <- c("survived","pclass","sex","age","sibsp","parch","fare","embarked")
missing <- setdiff(need, names(df))
if (length(missing)) {
  s(15, paste("Detected columns:", paste(names(df), collapse=", ")))
  stop(paste("Missing columns:", paste(missing, collapse=", ")))
}
df <- df[, need]

s(15, sprintf("Train shape: (%d, %d)", nrow(df), ncol(df)))
s(15, "Missing values (head):")
print(head(colSums(is.na(df))))

# ---- STEP 16: Feature engineering ----
# Impute simple medians / modes; ensure proper types
num_cols <- c("age","sibsp","parch","fare","pclass")
cat_cols <- c("sex","embarked")

# numeric impute
for (c in num_cols) {
  if (anyNA(df[[c]])) {
    med <- median(df[[c]], na.rm = TRUE)
    df[[c]][is.na(df[[c]])] <- med
    s(16, sprintf("Imputed median for %s = %.4f", c, med))
  }
}
# categorical impute
for (c in cat_cols) {
  if (anyNA(df[[c]])) {
    mode <- names(sort(table(df[[c]]), decreasing = TRUE))[1]
    df[[c]][is.na(df[[c]])] <- mode
    s(16, sprintf("Imputed mode for %s = %s", c, mode))
  }
}

# factors for GLM; Survived numeric 0/1
df <- df |>
  mutate(
    survived = as.integer(survived),
    sex = factor(sex),
    embarked = factor(embarked),
    pclass = as.integer(pclass)
  )

s(16, "Feature columns used: pclass, sex, age, sibsp, parch, fare, embarked")

# ---- STEP 17: Fit on full train and report training accuracy ----
s(17, "Fitting logistic regression (glm, binomial).")
glm_full <- glm(
  survived ~ pclass + sex + age + sibsp + parch + fare + embarked,
  data = df, family = binomial()
)
pred_train_prob <- as.numeric(predict(glm_full, type = "response"))
pred_train <- as.integer(pred_train_prob >= 0.5)
train_acc <- mean(pred_train == df$survived)
s(17, sprintf("Training accuracy: %.4f", train_acc))

# ---- STEP 18: Holdout validation (80/20 split on train) ----
set.seed(42)
idx <- sample(seq_len(nrow(df)), size = floor(0.8 * nrow(df)))
tr <- df[idx, , drop = FALSE]
te <- df[-idx, , drop = FALSE]
glm_hold <- glm(
  survived ~ pclass + sex + age + sibsp + parch + fare + embarked,
  data = tr, family = binomial()
)
pred_hold <- as.numeric(predict(glm_hold, newdata = te, type = "response"))
hold_acc <- mean(as.integer(pred_hold >= 0.5) == te$survived)
s(18, sprintf("Holdout accuracy: %.4f", hold_acc))

# ---- STEP 19: Persist model artifacts ----
model_rds <- file.path(art_dir, "model_r_glm.rds")
saveRDS(glm_hold, model_rds)
s(19, sprintf("Saved R model to: %s", model_rds))

# ---- STEP 20: Predict test.csv (if present) and export submission ----
if (file.exists(test_path)) {
  s(20, "Loading Kaggle test.csv and generating submission.")
  test <- read_csv_robust(test_path)
  names(test) <- norm_names(names(test))
  # keep needed features, impute using train med/mode
  need_te <- c("pclass","sex","age","sibsp","parch","fare","embarked","passengerid")
  miss_te <- setdiff(need_te, names(test))
  if (length(miss_te)) {
    s(20, paste("Detected test columns:", paste(names(test), collapse=", ")))
    stop(paste("Missing test columns:", paste(miss_te, collapse=", ")))
  }
  test <- test[, need_te]
  
  # impute using train statistics
  # numeric
  for (c in c("age","sibsp","parch","fare","pclass")) {
    med <- median(df[[c]], na.rm = TRUE)
    test[[c]][is.na(test[[c]])] <- med
  }
  # categorical
  for (c in c("sex","embarked")) {
    mode <- names(sort(table(df[[c]]), decreasing = TRUE))[1]
    test[[c]][is.na(test[[c]])] <- mode
  }
  
  test <- test |>
    mutate(
      sex = factor(sex, levels = levels(df$sex)),
      embarked = factor(embarked, levels = levels(df$embarked)),
      pclass = as.integer(pclass)
    )
  
  prob <- as.numeric(predict(glm_hold, newdata = test, type = "response"))
  pred <- as.integer(prob >= 0.5)
  submission <- tibble(PassengerId = test$passengerid, Survived = pred)
  sub_path <- file.path(art_dir, "submission_r.csv")
  readr::write_csv(submission, sub_path)
  s(20, sprintf("Wrote submission: %s (Note: official test has no labels)", sub_path))
} else {
  s(20, "No test.csv detected; skipping submission.")
}

# ---- STEP 21: Summary ----
cat("\n[SUMMARY - R]\n")
cat(sprintf("  Train accuracy  : %.4f\n", train_acc))
cat(sprintf("  Holdout accuracy: %.4f\n", hold_acc))
cat(sprintf("  Model path      : %s\n", model_rds))
if (file.exists(file.path(art_dir, "submission_r.csv"))) {
  cat(sprintf("  Submission path : %s\n", file.path(art_dir, "submission_r.csv")))
}

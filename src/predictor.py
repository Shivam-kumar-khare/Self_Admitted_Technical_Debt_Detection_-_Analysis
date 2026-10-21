import os
import re
import warnings
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.naive_bayes import MultinomialNB


class SATDPredictor:

  def __init__(
      self,
      baseline_csv_path="technical_debt_dataset_original.csv",
      ratio=0.1,
  ):
    self.baseline_csv_path = baseline_csv_path
    self.ratio = ratio
    self.stemmer = PorterStemmer()
    self.models_pipeline = []
    self.excluded_projects = ["apache-ant-1.7.0", "emf-2.4.1"]

  def _stemmed_tokenizer(self, doc):
    """Custom tokenizer applying Porter stemming matching the notebook logic."""
    tokens = re.findall(r"(?u)\b\w\w+\b", str(doc).lower())
    return [self.stemmer.stem(t) for t in tokens]

  def train_ensemble(self):
    """Loads baseline data, applies missing value checks, target mapping, column dropping,

    and dictionary structuring exactly as done in the notebook, then trains the ensemble.
    """
    print("Loading baseline dataset for cross-project training...")
    df = pd.read_csv(self.baseline_csv_path)

    # 1. Missing value analysis and verification
    print("Missing values in baseline dataset:\n", df.isnull().sum())

    # 2. Dictionary structuring by project name
    dfs = {
        project: group.copy().reset_index(drop=True)
        for project, group in df.groupby("projectname")
    }

    for p in self.excluded_projects:
      dfs.pop(p, None)

    # 3. Label creation (Target mapping) & 4. Dropping extra columns
    for project_name, group_df in dfs.items():
      group_df["target"] = group_df["classification"].apply(
          lambda t: 0 if str(t).upper() == "WITHOUT_CLASSIFICATION" else 1
      )
      group_df.drop(
          columns=["classification", "projectname"],
          inplace=True,
          errors="ignore",
      )

    print(
        f"Training ensemble across {len(dfs)} baseline projects: "
        f"{list(dfs.keys())}"
    )

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")

      for source_name, train_df in dfs.items():
        X_text = train_df["commenttext"].fillna("").values
        y_train = train_df["target"].values.astype(int)

        # 1. Text Vectorization with Porter stemming
        vectorizer = TfidfVectorizer(
            tokenizer=self._stemmed_tokenizer, token_pattern=None
        )
        X_train = vectorizer.fit_transform(X_text)

        # 2. Information Gain Feature Selection
        percentile_val = int(self.ratio * 100)
        selector = SelectPercentile(
            score_func=mutual_info_classif, percentile=percentile_val
        )
        X_train_selected = selector.fit_transform(X_train, y_train)

        # 3. SMOTE Over-sampling
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_selected, y_train
        )

        # 4. Model Training (Naive Bayes)
        clf = MultinomialNB()
        clf.fit(X_train_resampled, y_train_resampled)

        self.models_pipeline.append({
            "source": source_name,
            "vectorizer": vectorizer,
            "selector": selector,
            "clf": clf,
        })

    print(
        "Ensemble training complete! Total models in ensemble:"
        f" {len(self.models_pipeline)}"
    )

  def predict_snapshot(self, raw_comments_csv_path, output_labeled_csv_path):
    """Runs ensemble prediction on a Spring version snapshot CSV."""
    if not self.models_pipeline:
      raise ValueError(
          "Model ensemble has not been trained yet. Call train_ensemble() first."
      )

    print(f"Loading raw comments from {raw_comments_csv_path}...")
    test_df = pd.read_csv(raw_comments_csv_path)

    if len(test_df) == 0:
      print("Warning: The input CSV is empty. Skipping prediction.")
      return

    # Handle column name variance ('comment_text' from extractor vs 'commenttext' from baseline)
    text_col = "comment_text" if "comment_text" in test_df.columns else "commenttext"
    X_test_text = test_df[text_col].fillna("").values
    votes = np.zeros(len(test_df))

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")

      for model_info in self.models_pipeline:
        vectorizer = model_info["vectorizer"]
        selector = model_info["selector"]
        clf = model_info["clf"]

        X_test = vectorizer.transform(X_test_text)
        X_test_selected = selector.transform(X_test)
        preds = clf.predict(X_test_selected)

        source_votes = np.where(preds == 1, 1, -1)
        votes += source_votes

    # Majority voting threshold (votes >= 0)
    final_preds = np.where(votes >= 0, 1, 0)
    test_df["predicted_target"] = final_preds

    os.makedirs(os.path.dirname(output_labeled_csv_path), exist_ok=True)
    test_df.to_csv(output_labeled_csv_path, index=False, encoding="utf-8")

    satd_count = sum(final_preds)
    print(
        f"Prediction complete. Labeled {len(test_df)} comments"
        f" ({satd_count} classified as SATD)."
    )
    print(f"Saved to {output_labeled_csv_path}")


# --- Quick Test Snippet ---
if __name__ == "__main__":
  predictor = SATDPredictor(
      baseline_csv_path="technical_debt_dataset_original.csv", ratio=0.1
  )
  predictor.train_ensemble()

  predictor.predict_snapshot(
      raw_comments_csv_path="./data/raw_comments/v5.3.0.csv",
      output_labeled_csv_path="./data/satd_labled/v5.3.0_satd.csv",
  )
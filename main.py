import os
import sys
import time

# Add src directory to system path
sys.path.append(os.path.abspath("./src"))

from repo_manager import SpringRepoManager
from extractor import JavaCommentExtractor
from predictor import SATDPredictor


def run_pipeline():
  print("=" * 60)
  print("STARTING SATD LONGITUDINAL EVOLUTION PIPELINE (SPRING 5.3)")
  print("=" * 60)

  start_time = time.time()

  # ---------------------------------------------------------
  # Phase 1: Repository Management & Tag Discovery
  # ---------------------------------------------------------
  print("\n--- PHASE 1: REPOSITORY MANAGEMENT ---")
  repo_mgr = SpringRepoManager()
  tags = repo_mgr.get_v53_tags()
  print(f"Discovered {len(tags)} stable v5.3 patch releases.")

  if not tags:
    print("Error: No tags found. Exiting pipeline.")
    return

  # ---------------------------------------------------------
  # Phase 3 Pre-requisite: Ensemble Training (Train Once)
  # ---------------------------------------------------------
  print("\n--- PHASE 3 SETUP: TRAINING CROSS-PROJECT ENSEMBLE ---")
  predictor = SATDPredictor(
      baseline_csv_path="technical_debt_dataset_original.csv", ratio=0.1
  )
  predictor.train_ensemble()

  # Initialize Extractor
  extractor = JavaCommentExtractor(repo_dir=repo_mgr.repo_dir)

  # Output Directories
  raw_dir = "./data/raw_comments"
  labeled_dir = "./data/satd_labled"
  os.makedirs(raw_dir, exist_ok=True)
  os.makedirs(labeled_dir, exist_ok=True)

  # ---------------------------------------------------------
  # Phases 2 & 3 Automated Loop Across All 40 Patches
  # ---------------------------------------------------------
  print("\n--- EXECUTING BATCH PROCESSING LOOP ACROSS 40 PATCHES ---")

  for idx, tag in enumerate(tags, 1):
    print(f"\n[{idx}/{len(tags)}] Processing Patch: {tag}")
    print("-" * 40)

    # File paths for this patch
    raw_csv = os.path.join(raw_dir, f"{tag}.csv")
    labeled_csv = os.path.join(labeled_dir, f"{tag}_satd.csv")

    try:
      # Step A: Checkout Tag
      repo_mgr.checkout_version(tag)

      # Step B: Extraction (skip if raw CSV already exists)
      if not os.path.exists(raw_csv):
        extractor.extract_version_comments(tag, raw_csv)
      else:
        print(f"Raw comments already extracted for {tag}. Skipping extraction.")

      # Step C: Predict / Infer SATD
      if not os.path.exists(labeled_csv):
        predictor.predict_snapshot(
            raw_comments_csv_path=raw_csv, output_labeled_csv_path=labeled_csv
        )
      else:
        print(f"SATD predictions already exist for {tag}. Skipping inference.")

    except Exception as e:
      print(f"Error processing patch {tag}: {e}")
      continue

  elapsed_min = (time.time() - start_time) / 60
  print("\n" + "=" * 60)
  print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_min:.2f} MINUTES!")
  print("=" * 60)


if __name__ == "__main__":
  run_pipeline()
import os
import re
import pandas as pd


class SATDDeltaTracker:

  def __init__(self, labeled_dir="./data/satd_labled"):
    self.labeled_dir = labeled_dir

  def get_sorted_tags(self):
    """Discovers and sorts all v5.3.x labeled CSV files chronologically."""
    files = os.listdir(self.labeled_dir)
    tags = []
    for f in files:
      if f.startswith("v5.3.") and f.endswith("_satd.csv"):
        tag = f.replace("_satd.csv", "")
        tags.append(tag)

    # Sort numerically by patch number (e.g., v5.3.9 comes before v5.3.10)
    def patch_key(t):
      return int(t.split(".")[2])

    return sorted(tags, key=patch_key)

  def track_evolution(self, output_summary_path="./outputs/satd_evolution_summary.csv"):
    """Tracks SATD counts, introductions, and resolutions across consecutive patches."""
    tags = self.get_sorted_tags()
    if not tags:
      print("Error: No labeled dataset found in", self.labeled_dir)
      return

    print(f"Tracking delta evolution across {len(tags)} patch releases...")
    records = []

    prev_satd_hashes = set()

    for idx, tag in enumerate(tags):
      file_path = os.path.join(self.labeled_dir, f"{tag}_satd.csv")
      df = pd.read_csv(file_path)

      # Filter only predicted SATD rows (predicted_target == 1)
      satd_df = df[df["predicted_target"] == 1]
      current_satd_hashes = set(satd_df["comment_hash"].values)
      total_comments = len(df)
      total_satd = len(satd_df)

      if idx == 0:
        # Baseline version (v5.3.0) has no preceding version to compare deltas
        introduced = 0
        resolved = 0
      else:
        # Introduced: present now, but wasn't in previous version
        introduced = len(current_satd_hashes - prev_satd_hashes)
        # Resolved: was in previous version, but no longer present
        resolved = len(prev_satd_hashes - current_satd_hashes)

      net_change = introduced - resolved

      records.append({
          "version": tag,
          "total_comments": total_comments,
          "satd_count": total_satd,
          "satd_percentage": round((total_satd / total_comments) * 100, 2) if total_comments > 0 else 0,
          "introduced": introduced,
          "resolved": resolved,
          "net_change": net_change,
      })

      # Update previous hashes for next iteration
      prev_satd_hashes = current_satd_hashes
      print(f"Processed {tag} -> SATD: {total_satd} | Introduced: +{introduced} | Resolved: -{resolved}")

    # Save summary report
    summary_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)
    summary_df.to_csv(output_summary_path, index=False)
    print(f"\nDelta tracking complete! Summary saved to {output_summary_path}")
    return summary_df


if __name__ == "__main__":
  tracker = SATDDeltaTracker()
  summary = tracker.track_evolution()
  print("\nFirst 5 rows of evolution summary:")
  print(summary.head())
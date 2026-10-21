import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SATDVisualizer:

  def __init__(
      self,
      summary_path="./outputs/satd_evolution_summary.csv",
      labeled_dir="./data/satd_labled",
      output_dir="./outputs",
  ):
    self.summary_path = summary_path
    self.labeled_dir = labeled_dir
    self.output_dir = output_dir

  def generate_all_charts(self):
    if not os.path.exists(self.summary_path):
      print(f"Error: Summary file not found at {self.summary_path}. Run tracker.py first.")
      return

    df = pd.read_csv(self.summary_path)
    os.makedirs(self.output_dir, exist_ok=True)

    # Use clean plotting style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    versions = df["version"]
    tick_indices = range(0, len(versions), max(1, len(versions) // 15))
    tick_labels = [versions[i] for i in tick_indices]

    # ---------------------------------------------------------
    # Chart 1: Total SATD Count Trend
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(versions, df["satd_count"], color="royalblue", marker="o", linewidth=2, label="Total SATD")
    ax.set_xlabel("Spring 5.3 Patch Releases", fontweight="bold", fontsize=12)
    ax.set_ylabel("Total SATD Comments", color="royalblue", fontweight="bold", fontsize=12)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_title("Longitudinal Evolution of Self-Admitted Technical Debt (Spring 5.3)", fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(self.output_dir, "satd_total_trend.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Chart 2: Churn Bar Chart (Introduced vs Resolved)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(versions))
    width = 0.4
    ax.bar([i - width/2 for i in x], df["introduced"], width=width, color="crimson", label="Introduced (+)")
    ax.bar([i + width/2 for i in x], df["resolved"], width=width, color="forestgreen", label="Resolved (-)")
    ax.set_xlabel("Spring 5.3 Patch Releases", fontweight="bold", fontsize=12)
    ax.set_ylabel("Comment Count", fontweight="bold", fontsize=12)
    ax.set_title("SATD Churn: Introductions vs. Resolutions Across Patches", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(self.output_dir, "satd_churn_bar.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Chart 3: Net Debt Accumulation Area Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    net_changes = df["net_change"]
    ax.fill_between(versions, net_changes, color="darkorange", alpha=0.4)
    ax.plot(versions, net_changes, color="chocolate", marker="o", linewidth=1.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Spring 5.3 Patch Releases", fontweight="bold", fontsize=12)
    ax.set_ylabel("Net Debt Change (Added - Removed)", fontweight="bold", fontsize=12)
    ax.set_title("Net Technical Debt Growth / Reduction per Patch Release", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=90)
    fig.tight_layout()
    plt.savefig(os.path.join(self.output_dir, "satd_net_accumulation.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Chart 4: Sub-Module Heatmap
    # ---------------------------------------------------------
    print("Generating Spring sub-module heatmap from labeled CSVs...")
    module_data = []

    for tag in versions:
      file_path = os.path.join(self.labeled_dir, f"{tag}_satd.csv")
      if not os.path.exists(file_path):
        continue
      patch_df = pd.read_csv(file_path)
      satd_subset = patch_df[patch_df["predicted_target"] == 1]

      # Extract top-level module folder from file_path (e.g., spring-core, spring-web)
      def extract_module(path):
        parts = str(path).replace("\\", "/").split("/")
        return parts[0] if len(parts) > 0 else "unknown"

      satd_subset["module"] = satd_subset["file_path"].apply(extract_module)
      counts = satd_subset["module"].value_counts()

      for mod, count in counts.items():
        module_data.append({"version": tag, "module": mod, "satd_count": count})

    mod_df = pd.DataFrame(module_data)
    if not mod_df.empty:
      pivot_df = mod_df.pivot(index="module", columns="version", values="satd_count").fillna(0)
      # Filter for top modules to keep heatmap clean
      top_modules = pivot_df.sum(axis=1).nlargest(10).index
      pivot_filtered = pivot_df.loc[top_modules]

      fig, ax = plt.subplots(figsize=(14, 8))
      cax = ax.imshow(pivot_filtered, cmap="YlOrRd", aspect="auto")
      fig.colorbar(cax, label="SATD Comment Count")

      ax.set_xticks(range(len(pivot_filtered.columns)))
      ax.set_xticklabels(pivot_filtered.columns, rotation=90, fontsize=8)
      ax.set_yticks(range(len(pivot_filtered.index)))
      ax.set_yticklabels(pivot_filtered.index, fontsize=10)
      ax.set_xlabel("Spring 5.3 Patch Releases", fontweight="bold", fontsize=12)
      ax.set_ylabel("Spring Sub-Modules", fontweight="bold", fontsize=12)
      ax.set_title("Technical Debt Concentration Heatmap Across Spring Sub-Modules", fontsize=14, fontweight="bold", pad=15)

      fig.tight_layout()
      plt.savefig(os.path.join(self.output_dir, "satd_module_heatmap.png"), dpi=300)
      plt.close()

    print(f"All visualizations successfully generated and saved to {self.output_dir}/")


if __name__ == "__main__":
  visualizer = SATDVisualizer()
  visualizer.generate_all_charts()
#! 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


df = pd.read_csv("rc_rag_benchmark.csv")


# Output directory
os.makedirs("figures", exist_ok=True)

# ---------------------
# 1. Average Latency per Model
# ---------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Model", y="Latency", errorbar="sd")
plt.title("Average Latency by Model")
plt.ylabel("Latency (seconds)")
plt.tight_layout()
plt.savefig("figures/avg_latency.png", dpi=200)
plt.close()

# ---------------------
# 2. Latency Distribution (Boxplot)
# ---------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Model", y="Latency")
plt.title("Latency Distribution per Model")
plt.tight_layout()
plt.savefig("figures/latency_boxplot.png", dpi=200)
plt.close()


# ---------------------
# 4. Per-Query Latency Heatmap
# ---------------------
pivot = df.pivot_table(
    index="Query",
    columns="Model",
    values="Latency"
)

plt.figure(figsize=(10, 8))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
plt.title("Latency Heatmap by Query and Model")
plt.tight_layout()
plt.savefig("figures/latency_heatmap.png", dpi=200)
plt.close()


print("\nAll visualizations generated and saved in ./figures/")

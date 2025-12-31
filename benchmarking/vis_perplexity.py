import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("rag_evaluation_results.csv")

plt.figure(figsize=(7,6))
sns.scatterplot(data=df, x="semantic_similarity", y="latency")
plt.title("Semantic Similarity vs Latency")
plt.xlabel("Semantic Similarity")
plt.ylabel("Latency (sec)")
plt.grid(True)

plt.savefig("figures/scatterplot.png", dpi=200)
plt.show()



plt.close()
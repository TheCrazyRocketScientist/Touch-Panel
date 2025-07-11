import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data from your image (5×5 matrix)
data = [
    [125, 3, 0, 2, 0],
    [12, 60, 28, 29, 20],
    [14, 22, 60, 23, 15],
    [11, 23, 31, 51, 21],
    [13, 26, 19, 26, 30]
]

labels = [0, 1, 2, 3, 4]

# Set up figure for poster-quality export
plt.figure(figsize=(10, 8))  # Larger for higher clarity at 330 DPI

ax = sns.heatmap(
    data,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar=True,
    annot_kws={"size": 15}
)

plt.title("Confusion Matrix on Test Set", fontsize=18)
plt.xlabel("Predicted label", fontsize=18)
plt.ylabel("True label", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Save as transparent PNG at 330 DPI
plt.savefig("confusion_matrix_poster_new.png", dpi=330, transparent=True)
plt.close()

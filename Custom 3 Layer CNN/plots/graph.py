import matplotlib.pyplot as plt
import numpy as np

# Data from your table
classes = ["Apple", "Banana", "Orange", "Mixed"]
original_counts = [75, 73, 72, 20]
cleaned_counts = [75, 67, 72, 20]
final_counts = [75, 67, 72, 70]

# Bar width and positions
bar_width = 0.25
x = np.arange(len(classes))

# Plotting
plt.figure(figsize=(10, 6))

plt.bar(
    x - bar_width, original_counts, width=bar_width, label="Original", color="#FFA500"
)  # Normal orange
plt.bar(
    x, cleaned_counts, width=bar_width, label="After Cleaning", color="#FFB347"
)  # Lighter orange
plt.bar(
    x + bar_width,
    final_counts,
    width=bar_width,
    label="Final (Balanced)",
    color="#CC8400",
)  # Darker orange

# Labels and titles
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.title("Class Distribution Before and After Cleaning & Balancing")
plt.xticks(x, classes)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

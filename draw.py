import matplotlib.pyplot as plt
import numpy as np

# ResNet models and their corresponding R1 values
models = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
r1_values = [32, 39, 36, 44, 49.3]

# X-axis positions
x = np.arange(len(models))

# Set figure size to match 1143x481 pixels with DPI 100
figsize = (7.62, 3.21)  # Width and height in inches
dpi = 150  # DPI

# Create the bar chart
plt.figure(figsize=figsize, dpi=dpi)
plt.bar(x, r1_values, color="orange")

# Add labels and title
# plt.xlabel("ResNet Models")
plt.ylabel("R1 Accuracy")
# plt.title("ResNet Performance Comparison")
plt.xticks(x, models)
plt.ylim(0, 70)

# Add grid in the background
plt.grid(True, linestyle="--", alpha=0.6, axis="y")

# Set white background
plt.gca().set_facecolor("white")
plt.gcf().set_facecolor("white")

# Save the figure with exact pixel dimensions
plt.savefig("resnet_r1_chart.png", dpi=dpi, bbox_inches="tight", facecolor="white")
plt.show()

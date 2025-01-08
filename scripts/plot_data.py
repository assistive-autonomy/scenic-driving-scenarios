import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Script to be used to plot data for analysis
PLOT_WITH_EXECUTION = False

if PLOT_WITH_EXECUTION:
    data = {
        'number_of_turns': [1, 2, 3, 4],
        'intersection': [4.545454545,	9.090909091,	6.818181818,	11.36363636],
        'bypassing': [5.263157895,	0,	10.52631579,	0],
        'pedestrian': [8.108108108,	8.108108108,	5.405405405,	8.108108108],
        'total': [6,	7,	7,	8],
    }
else:
    data = {
        'number_of_turns': [1, 2, 3, 4],
        'intersection': [4.545454545,	13.63636364,	4.545454545,	9.090909091],
        'bypassing': [5.263157895,	5.263157895,	5.263157895,	0],
        'pedestrian': [8.108108108,	13.51351351,	2.702702703,	5.405405405],
        'total': [6,	12,	4,	6],
    }

df = pd.DataFrame(data)

# chang to cumulative sum
df['intersection'] = df['intersection'].cumsum()
df['bypassing'] = df['bypassing'].cumsum()
df['pedestrian'] = df['pedestrian'].cumsum()
df['total'] = df['total'].cumsum()

# Plotting
sns.set_theme(style="ticks")
sns.color_palette("Set2")
plt.figure(figsize=(8, 4))

plt.plot(df['number_of_turns'], df['intersection'], marker='o', label='Intersection')
plt.plot(df['number_of_turns'], df['bypassing'], marker='o', label='Bypassing')
plt.plot(df['number_of_turns'], df['pedestrian'], marker='o', label='Pedestrian')
plt.plot(df['number_of_turns'], df['total'], marker='o', label='Total')

plt.xlabel('Number of dialoge turns', fontsize=12, labelpad=10)
plt.ylabel('Succesful Scenario Generation (%)', fontsize=12)
plt.xticks(df['number_of_turns'].unique())
ax = plt.gca()
ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
# Use tight layout to add padding
plt.tight_layout()
plt.legend(title='Scenario Type', ncols=2)
plt.show()

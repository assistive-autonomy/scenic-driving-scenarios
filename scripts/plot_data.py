import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Script to be used to plot data for analysis
data = {
    'trial_number': range(1, 121),
    'number_of_turns': np.random.choice([1, 2, 3, 4, 5], size=120),
    'success': np.random.choice([0, 1], size=120)
}

df = pd.DataFrame(data)

# Sort the dataframe by 'trial_number'
df = df.sort_values(by='trial_number')

# Calculate cumulative success rate
grouped = df.groupby('number_of_turns')['success']

df['cumulative_success'] = grouped.cumsum() / (grouped.cumcount() + 1)

# Plotting
sns.set_theme(style="ticks")
sns.color_palette("Set2")
plt.figure(figsize=(8, 4))

sns.lineplot(x='number_of_turns', y='cumulative_success', data=df, marker='o', errorbar="sd", err_style="band")
plt.title('Cumulative Success Rate by Number of Turns', fontsize=14)
plt.xlabel('Number of Turns', fontsize=12)
plt.ylabel('Cumulative Success Rate', fontsize=12)
plt.xticks(df['number_of_turns'].unique())
plt.yticks([0, 0.25, 0.5, 0.75, 1])
ax = plt.gca()
ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
plt.show()

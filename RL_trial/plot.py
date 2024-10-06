import matplotlib.pyplot as plt
import pandas as pd

# Read the data from the file
data = pd.read_csv('reward_0050.txt')

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(data['Episode'], data['Training_Reward'], label='Training Reward', marker='o')
plt.plot(data['Episode'], data['Validation_Reward'], label='Validation Reward', marker='s')

# Customize the plot
plt.title('Training and Validation Rewards over Episodes')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
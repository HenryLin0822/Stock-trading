import numpy as np
import matplotlib.pyplot as plt

def parse_results_file(filename):
    episodes = []
    training_returns = []
    validation_returns = []
    
    with open(filename, 'r') as f:
        next(f)  # Skip the header line
        for line in f:
            parts = line.strip().split(',')
            episodes.append(int(parts[0]))
            training_returns.append(float(parts[1]))
            validation_returns.append(float(parts[2]))
    
    return np.array(episodes), np.array(training_returns), np.array(validation_returns)

def calculate_stats(returns, e_start, e_finish):
    subset = returns[e_start:e_finish+1]  # +1 to include e_finish
    return np.mean(subset), np.std(subset)

def plot_results(episodes, training_returns, validation_returns, n, stock, gamma):
    plt.figure(figsize=(12, 6))
    plt.plot(episodes[:n], training_returns[:n], label='Training Return')
    plt.plot(episodes[:n], validation_returns[:n], label='Validation Return')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Training and Validation Returns')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/results_'+stock+'_'+str(gamma)+'.png')
    plt.close()

def main(e_start, e_finish, n, stock, gamma):
    filename = 'results/results_'+stock+'_'+str(gamma)+'.txt'
    episodes, training_returns, validation_returns = parse_results_file(filename)
    
    train_mean, train_std = calculate_stats(training_returns, e_start, e_finish)
    val_mean, val_std = calculate_stats(validation_returns, e_start, e_finish)
    
    print(f"Training Return (episodes {e_start}-{e_finish}):")
    print(f"  Mean: {train_mean:.4f}")
    print(f"  Std:  {train_std:.4f}")
    print(f"Validation Return (episodes {e_start}-{e_finish}):")
    print(f"  Mean: {val_mean:.4f}")
    print(f"  Std:  {val_std:.4f}")
    
    plot_results(episodes, training_returns, validation_returns, n, stock, gamma)
    print(f"Plot saved.")

if __name__ == "__main__":
    e_start = 200  # Example: start from episode 5
    e_finish = 300  # Example: end at episode 10
    n = 300  # Example: plot all 14 episodes
    stock = '2303'
    gamma = 0.0
    main(e_start, e_finish, n, stock, gamma)
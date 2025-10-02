import gymnasium as gym
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from dqnagent import DQNAgent  # your agent code
from qnetwork import QNetwork  # QNetwork class
import os

# ✅ Exploration values to test
exploration_values = [0.0, 0.25, 0.5]

# 1️⃣ Initialize environment
env = gym.make("CartPole-v1", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 2️⃣ Load agent
agent = DQNAgent(state_size, action_size)
agent.q_network.load_state_dict(torch.load("best_cartpole.pth"))
agent.q_network.eval()  # evaluation mode

# 3️⃣ Evaluation parameters
num_episodes = 10
max_t = 500
threshold = 195

# 4️⃣ Loop through each exploration value
for eps in exploration_values:
    save_dir = f"evaluation_results/eps_{eps}"
    os.makedirs(save_dir, exist_ok=True)
    scores = []

    print(f"\n===== Evaluating EPS = {eps} =====\n")

    for i_episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0

        print(f"--- Episode {i_episode} ---")

        while not done and step < max_t:
            env.render()
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)

            # Choose action with epsilon (exploration)
            if np.random.rand() < eps:
                action = np.random.randint(action_size)  # random action
            else:
                action = torch.argmax(q_values).item()  # best action
                
             # state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # with torch.no_grad():
        #     q_values = agent.q_network(state_tensor)
        # action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward
            step += 1

            time.sleep(0.01)

        print(f"Episode Reward: {episode_reward}")
        scores.append(episode_reward)

    # Compute metrics
    avg_score = np.mean(scores)
    success_rate = np.sum(np.array(scores) >= threshold) / num_episodes * 100

    print(f"\n✅ EPS={eps} Evaluation Results:")
    print(f"Average Score over {num_episodes} episodes: {avg_score:.2f}")
    print(f"Success Rate (>={threshold}): {success_rate:.2f}%")

    # Save metrics to text file
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"EPS: {eps}\n")
        f.write(f"Scores: {scores}\n")
        f.write(f"Average Score: {avg_score:.2f}\n")
        f.write(f"Success Rate: {success_rate:.2f}%\n")

    # Plot rewards
    plt.figure(figsize=(10,5))
    plt.plot(range(1, num_episodes+1), scores, marker='o')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold {threshold}")
    plt.title(f"CartPole Episode Rewards (EPS={eps})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "episode_rewards.jpg"))
    plt.close()  # close to avoid overlapping plots

# 5️⃣ Close environment
env.close()

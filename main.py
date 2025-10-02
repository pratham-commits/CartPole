# main.py
import gymnasium as gym
import torch
import time
from dqnagent import DQNAgent
from qnetwork import QNetwork

def play_cartpole(trained_model_path="best_cartpole.pth", num_episodes=5, max_t=500):
    """
    Runs the trained DQN agent on CartPole-v1 to watch its gameplay.
    """
    # Initialize environment with visuals
    env = gym.make("CartPole-v1", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Load agent
    agent = DQNAgent(state_size, action_size)
    agent.q_network.load_state_dict(torch.load(trained_model_path))
    agent.q_network.eval()  # set to evaluation mode

    print(f"\n===== Watching the trained agent for {num_episodes} episodes =====\n")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        step = 0

        print(f"--- Episode {episode} ---")

        while not done and step < max_t:
            env.render()
            
            # Choose action (pure exploitation)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
            action = torch.argmax(q_values).item()

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            step += 1

            time.sleep(0.01)  # slow down for smooth visualization

    env.close()
    print("\n✅ Finished watching the agent.")

if __name__ == "__main__":
    play_cartpole(trained_model_path="best_cartpole.pth", num_episodes=5)

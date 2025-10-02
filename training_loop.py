# Initialize environment
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Create agent
agent = DQNAgent(state_size, action_size)

# Training hyperparameters
episodes = 2000
total_steps = 0
update_target_every = 1000  # steps
best_score = 0
scores = []

for e in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store experience
        agent.remember(state, action, reward, next_state, done)

        # Train agent
        agent.train()

        state = next_state
        total_reward += reward
        total_steps += 1

        # Update target network periodically
        if total_steps % update_target_every == 0:
            agent.update_target()

    # Decay epsilon
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    # Save best model
    if total_reward > best_score:
        torch.save(agent.q_network.state_dict(), "best_cartpole.pth")
        best_score = total_reward

    scores.append(total_reward)
    if len(scores) >= 100:
        moving_avg = np.mean(scores[-100:])
    else:
        moving_avg = np.mean(scores)

    print(f"Episode {e+1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}, Moving Avg: {moving_avg:.2f}")
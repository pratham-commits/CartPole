# CartPole DQN Agent

This repository contains a **Deep Q-Network (DQN) implementation** to solve the classic OpenAI Gym **CartPole-v1** environment using **PyTorch**. The agent learns to balance the pole on the cart by choosing actions (move left or right) based on its current state.

---

## 🧠 Project Overview

- **Environment**: `CartPole-v1` from OpenAI Gym
- **State Size**: 4 (cart position, cart velocity, pole angle, pole velocity at tip)
- **Action Size**: 2 (left, right)
- **Agent**: Deep Q-Network with target network
- **Experience Replay**: Stores past transitions to train the agent more efficiently
- **Exploration Strategy**: Epsilon-greedy
- **Training**: 2000 episodes, adaptive epsilon decay

---

## 📂 Repository Structure

```
.
├── qnetwork.py          # Defines QNetwork (DQN neural network)
├── replaybuffer.py      # Replay buffer for experience replay
├── dqnagent.py          # DQN agent (training and action selection)
├── train_cartpole.py    # Script to train the DQN agent
├── evaluate_cartpole.py # Script to evaluate the trained agent
├── best_cartpole.pth    # Trained model weights (after training)
├── evaluation_results/  # Stores evaluation metrics and plots
└── README.md
```

---

## ⚡ Installation

1. Clone the repo:

```bash
git clone <your-repo-url>
cd <repo-folder>
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

3. Install required packages:

```bash
pip install torch torchvision torchaudio
pip install gymnasium matplotlib numpy
```

---

## 🏋️ Training the Agent

To train the agent from scratch:

```bash
python train_cartpole.py
```

**Training highlights**:

- The agent interacts with the environment and stores experiences in the replay buffer.
- Q-network and target Q-network are updated periodically.
- Epsilon is decayed over time to balance exploration and exploitation.
- The best model weights are saved as `best_cartpole.pth`.
- Moving average of rewards is printed for each episode.

---

## 🎮 Evaluating the Agent

To watch the trained agent play the game:

```bash
python evaluate_cartpole.py
```

**Features**:

- Renders the CartPole environment in real-time.
- Evaluates for **multiple exploration values**: `0.0`, `0.25`, `0.5`.
- Saves **evaluation metrics** (average score, success rate) to a text file.
- Saves **reward plots** as `.jpg` inside `evaluation_results/eps_<value>/`.

---

## 🧩 Key Classes and Files

### 1. `QNetwork` (`qnetwork.py`)

- Multi-layer fully connected neural network
- Input: state vector
- Output: Q-values for each action
- Uses **ReLU** activations

### 2. `ReplayBuffer` (`replaybuffer.py`)

- Stores transitions `(state, action, reward, next_state, done)`
- Randomly samples mini-batches for training

### 3. `DQNAgent` (`dqnagent.py`)

- Implements the DQN algorithm:
  - `act(state)`: chooses an action (epsilon-greedy)
  - `train()`: updates Q-network using sampled experiences
  - `update_target()`: syncs target network with Q-network
  - `remember()`: stores transitions in replay buffer
- Hyperparameters:
  - `gamma = 0.99`, `epsilon = 1.0 → 0.01`, `epsilon_decay = 0.998`
  - `batch_size = 64`, `memory_capacity = 50,000`

---

## 📈 Results

After training, the agent is capable of:

- Balancing the pole consistently for ~200 steps or more.
- Handling the pole’s movement from either side.
- Performing well under different exploration scenarios.

**Example evaluation output**:

```
✅ EPS=0.0 Evaluation Results:
Average Score over 10 episodes: 225.30
Success Rate (>=195): 30.00%
```

Plots and metrics are saved automatically in `evaluation_results/`.

---

## 🚀 How to Use

1. Train the agent using `train_cartpole.py` or use pre-trained weights.
2. Evaluate using `evaluate_cartpole.py` to watch the agent play in real-time.
3. Check `evaluation_results/` for detailed performance metrics and reward plots.

---

## 🔧 Notes

- Tested on **Python 3.10**, **PyTorch 2.1+**, **Gymnasium 1.2+**.
- Designed for both **CPU** and **GPU** training.
- Visualization requires `render_mode="human"`; on Colab use `gymnasium` without rendering.

---

## 📌 References

- OpenAI Gym: [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- PyTorch DQN Tutorials
- Reinforcement Learning Theory: Sutton & Barto, *Reinforcement Learning: An Introduction*

---



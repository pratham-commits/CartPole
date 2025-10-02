from qnetwork import QNetwork
from replaybuffer import ReplayBuffer
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
  def __init__(self,state_size,action_size):
    self.state_size=state_size
    self.action_size=action_size

    # Defining the QNetwork and target network
    self.q_network=QNetwork(self.state_size,self.action_size)
    self.target_q_network=QNetwork(self.state_size,self.action_size)

    #copying the weight initially from the Qnetwork to the target network
    self.target_q_network.load_state_dict(self.q_network.state_dict())
        
    self.optimizer=optim.Adam(self.q_network.parameters(),lr=0.001)
    self.memory=ReplayBuffer(50000)

    # Hyperparams
    self.gamma=0.99
    self.epsilon=1.0
    self.epsilon_min=0.01
    self.epsilon_decay=0.998
    self.batch_size=64

  def remember(self,state,action,reward,next_state,done):
    self.memory.push(state,action,reward,next_state,done)


  def act(self,state):
    if np.random.rand() < self.epsilon:
      return random.randrange(self.action_size) # EXPLORE
    
    state=torch.FloatTensor(state).unsqueeze(0)
    q_values=self.q_network(state)
    return torch.argmax(q_values).item() # EXPLOIT : choose action with highest Q-value
  

  def train(self):
    if len(self.memory)<self.batch_size:
      return

    states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.BoolTensor(dones)

    # Current Q-values for chosen actions
    q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target Q-values using target network
    next_q_values = self.target_q_network(next_states).max(1)[0]
    target_q_values = rewards + self.gamma * next_q_values * (1 - dones.float())

    # compute loss
    loss=nn.MSELoss()(q_values,target_q_values.detach())

    # Backpropagation

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def update_target(self):
    self.target_q_network.load_state_dict(self.q_network.state_dict())
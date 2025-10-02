from collections import deque
import random
import numpy as np
class ReplayBuffer:
  def __init__(self,capacity=50000):
    self.buffer=deque(maxlen=capacity)

  def push(self,state,action,reward,next_state,done):
    self.buffer.append((state,action,reward,next_state,done))


# core function of the replay buffer , randomly sampling the stored experiences
  def sample(self,batch_size):
    batch=random.sample(self.buffer,batch_size)
    states,actions,rewards,next_states,dones=zip(*batch)
    return np.array(states),actions,rewards,np.array(next_states),dones

  def __len__(self):
    return len(self.buffer)
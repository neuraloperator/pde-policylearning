import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
MEMORY_SIZE = 100000
BATCH_SIZE = 64
TAU = 5e-3


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64) -> None:
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * 2
        return x
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64) -> None:
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class ReplayMemory:
    def __init__(self, capacity) -> None:
        self.buffer = deque(maxlen=capacity)
    
    def add_memo(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)
    

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        self.replay_buffer = ReplayMemory(MEMORY_SIZE)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self, ):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # update critic
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions.detach())  # [N, 1]
        target_Q = rewards + (GAMMA * target_Q * (1 - dones))
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # update actor
        actor_loss = - self.critic(states, self.actor(states)).mean()
        # \Delta Q(s, mu_theta) = \Delta mu_theta(s) \Delta Q(s, a)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # update target networks of critic and actor
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
             
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

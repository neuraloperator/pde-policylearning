import gymnasium as gym
import numpy as np
import random
import os
from agent_ddpg import DDPGAgent

# Initialize env
env = gym.make(id='Pendulum-v1')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]  # 1

# Hyperparameters
NUM_EPISODE = 100
NUM_STEP = 200
EPSILON_START = 1.6
EPSILON_END = 0.02
EPSILON_DECAY = 10000

agent = DDPGAgent(STATE_DIM, ACTION_DIM)

REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    episode_reward = 0
    for step_i in range(NUM_STEP):
        epsilon = np.interp(x=episode_i*NUM_STEP+step_i,xp=[0, EPSILON_DECAY], fp=[EPSILON_START, EPSILON_END])
        random_sample = random.random()
        if random_sample <= epsilon:
            action = np.random.uniform(low=-2, high=2, size=ACTION_DIM)
        else:
            action = agent.get_action(state)
            
        next_state, reward, done, truncation, info = env.step(action)
        
        agent.replay_buffer.add_memo(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        
        agent.update()
        if done:
            break
    REWARD_BUFFER[episode_i] = episode_reward
    print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}.")


# current_path = os.path.dirname(os.path.realpath(__file__))
# model = current_path + "/models/"


env.close()
import pdb; pdb.set_trace()
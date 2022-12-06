#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym
from gym import Env
from stable_baselines3.common.vec_env import DummyVecEnv,VecFrameStack
from stable_baselines3 import A2C,PPO


# In[ ]:


def make_env():
    return MaskedPole()
    return gym.make('CartPole-v1')


# In[ ]:


class MaskedPole(Env):
    def __init__(self):
        super().__init__()
        self.env=gym.make('CartPole-v1')
        self.action_space=self.env.action_space
        self.observation_space=self.env.observation_space
    def reset(self):
        obs=self.env.reset()
        # print(obs)
        obs[1]=0
        obs[3]=0
        return obs
    def step(self,action):
        obs, rewards, dones, info = self.env.step(action)
        # print(obs)
        obs[1]=0
        obs[3]=0
        return obs, rewards, dones, info
    def render(self,mode="human"):
        self.env.render()


# In[ ]:


# mp=MaskedPole()


# In[ ]:


# mp.reset()
# mp.step(0)
# mp.render()


# In[ ]:


# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
env = gym.make('CartPole-v1')
# Frame-stacking with 4 frames
# env = DummyVecEnv([make_env])
# env = VecFrameStack(env, n_stack=4)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=40000)
obs = env.reset()
# model.save("A2C_breakout")
episodes=0
rew=0
while episodes<100:
    episodes+=1
    done=False
    env.reset()
    while not done:
        action, obs = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        # print(rewards,done)
        rew+=rewards
    # env.render()
print(rew/100)


# In[ ]:


episodes=0
rew=0
while episodes<100:
    episodes+=1
    done=False
    env.reset()
    while not done:
        action, obs = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        # print(rewards,done)
        rew+=rewards
    # env.render()
print(rew/100)


# In[ ]:


rewards+1


# In[ ]:


obs[:,0]=0


# In[ ]:


obs


# In[ ]:





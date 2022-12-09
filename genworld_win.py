#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
from gym import spaces
from gym import Env
import random
import numpy as np
from threading import Thread
import threading
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecFrameStack,StackedObservations


# In[2]:


# for thread in threading.enumerate(): 
#     print(thread.name)


# In[3]:


# for thread in threading.enumerate(): 
#     print(thread.name)


# ### Agent-based RL in Simple Worlds with windowing and Meta-RL
# 
# - using window of states in case where velocity is masked
# - can use meta-RL: **TBD test with varying physics in a CL setting**

# In[4]:


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


# In[5]:


# env = gym.make("CartPole-v1")
# env = gym.make('MountainCar-v0')


# In[6]:


env = MaskedPole()


# In[7]:


import import_ipynb
from aiagentbase import AIAgent,Controller,Memory,Perception,Actor


# In[8]:


class GenWorld():
    def __init__(self,env):
        self.env=env
        self.test_episodes=[]
        self.world_over=False
    def stop(self):
        self.world_over=True
    def run(self,agent=None,n_episodes=10,episode_maxlen=10):
        agent.observation_space=env.observation_space
        if 'training' not in agent.__dict__: agent.training=False
        if agent.training: testing=False 
        else: testing=True
        if agent.training: print('Starting Training time: ',agent.time)
        for episode in range(n_episodes):
            # print('CartAgent','starting episode')
            state=self.env.reset()
            agent.begin()
            # print(agent.time)#,agent.ep)
            for t in range(episode_maxlen):
                # env.render(mode='rgb_array')
                action=agent.act(state)
                # print(episode,t,'Action: ', action)
                state, reward, done, info = env.step(action)
                agent.reward((reward,done,info))
                # print(episode,t,'Reward sent: ', reward)
                if done:
                    break
            if self.world_over:break
            if not agent.training: self.test_episodes+=[episode]
            if not agent.training and not testing: 
                print('Training Over at time: ',agent.time)
                testing=True
        print('Testing Done time: ', agent.time, ' Reward: ', agent.avg_rew())
        return agent.avg_rew()


# In[9]:


#Doesnt use AIAgent Architecture Classes but implements the same interface - for initial testing
class RandomAgent():
    def __init__(self,action_space):
        self.action_space=action_space
        self.tot_rew=0
        self.rewL=[]
    def act(self,state):
        action = self.action_space.sample()
        return action
    def reward(self,rew):
        self.tot_rew+=rew[0]
    def begin(self,state):
        self.rewL+=[self.tot_rew]
    def avg_rew(self):
        return sum(self.rewL)/len(self.rewL)


# In[10]:


class RandomAIAgent(AIAgent):
    def __init__(self,action_space):
        super().__init__()
        self.actor=self.Actor(parent=self)
        self.action_space=action_space
        self.tot_rew=0
        self.rewL=[]
        
    class Actor(Actor):
        def __init__(self,parent): 
            super().__init__(parent=parent)
        def call_model(self,state):
        ##Overriding AIAgent.Model
            action = self.parent.action_space.sample()
            return action
        def compute_reward(self,reward):
            return reward[0]
    
    def reward(self,rew):
        ##Augmenting AIAgent
        self.tot_rew+=rew[0]
        return super().reward(rew)
    def begin(self):
        ##Augmenting AIAgent
        self.rewL+=[self.tot_rew]
        super().begin()
    def avg_rew(self):
        return sum(self.rewL)/len(self.rewL)


# In[11]:


agent=RandomAIAgent(env.action_space)
agent.training=False


# In[12]:


agent.debug=False
agent.use_memory=True


# In[13]:


agent.limit_memory=True
agent.memory.limit_perceptual=2
agent.memory.limit_sar=4


# In[14]:


world=GenWorld(env=env)


# In[15]:


agent.tot_rew,agent.rewL,agent.ep=0,[],[]


# In[16]:


worldthread=Thread(name='world',target=world.run,args=(agent,1000,200))


# In[17]:


worldthread.start()


# In[18]:


agent.avg_rew()/len(agent.ep)


# In[19]:


# world.run(agent,10,10)


# In[20]:


# agent.memory.perceptual_memory


# ### Training an AI Agent's Model using Generic RL Agent

# In[21]:


from threading import Thread
import threading
import sys


# In[22]:


from queue import Queue


# In[23]:


from aiagentbase import RLAgent


# In[24]:


training_steps=30000


# In[25]:


agent=RLAgent(algoclass=PPO,action_space=env.action_space,observation_space=env.observation_space,
              verbose=1,win=4,soclass=StackedObservations,metarl=False)


# In[26]:


agent.debug=False
agent.use_memory=True
agent.training=True


# In[27]:


agent.rewL=[]
agent.tot_rew=0
agent.ep=[]


# In[28]:


if agent.training: agent.start(training_steps=training_steps)


# In[29]:


world=GenWorld(env=env)


# In[30]:


# worldthread=Thread(name='world',target=world.run,args=(agent,2000,200))


# In[31]:


# worldthread.start()


# In[32]:


world.run(agent,n_episodes=2000,episode_maxlen=200)


# In[33]:


# agent.tot_rew/len(agent.ep)


# In[34]:


# len(agent.logL)


# In[35]:


# from matplotlib import pyplot as plt


# In[36]:


# testing_len=len([agent.rewL[t] for t in world.test_episodes])


# In[37]:


# testing_len


# In[38]:


# agent.rewL


# In[48]:


print(np.gradient(agent.rewL).mean())


# In[40]:


# plt.plot(np.gradient(agent.rewL))


# In[51]:


episodes = 500
rewL=[]
agent.training=False
for episode in range(1, episodes+1):
    done = False
    score = 0 
    steps=0
    state = env.reset()
    while not done and steps<=200:
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        score+=reward
        steps+=1
    # print('Episode:{} Score:{}'.format(episode, score))
    rewL+=[score]
env.close()


# In[45]:


# from matplotlib import pyplot as plt
# import numpy as np


# In[53]:


print(np.array(rewL).mean())


# In[ ]:


# plt.plot(rewL)


# In[ ]:


# PPO??


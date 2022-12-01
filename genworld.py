#!/usr/bin/env python
# coding: utf-8

# In[1]:
import warnings
warnings.simplefilter("ignore")

import gym
from gym import spaces
import random
import numpy as np
from threading import Thread
import threading
from stable_baselines3 import PPO


# In[2]:


# for thread in threading.enumerate(): 
#     print(thread.name)


# In[3]:


# for thread in threading.enumerate(): 
#     print(thread.name)


# ### Agent-based RL in Simple Worlds

# In[4]:


env = gym.make("CartPole-v1")
# env = gym.make('MountainCar-v0')


# In[5]:


import import_ipynb
from aiagentbase import AIAgent,Controller,Memory,Perception,Actor


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


agent=RandomAIAgent(env.action_space)
agent.training=False


# In[10]:


agent.debug=False
agent.use_memory=True


# In[11]:


world=GenWorld(env=env)


# In[12]:


worldthread=Thread(name='world',target=world.run,args=(agent,1000,200))


# In[13]:


worldthread.start()


# In[14]:


# world.run(agent,10,10)


# In[15]:


print(agent.memory.sar_memory)



# ### Training an AI Agent's Model using Generic RL Agent

# In[16]:


from threading import Thread
import threading
import sys


# In[17]:


from aiagentbase import RLAgent


# In[18]:


training_steps=2048


# In[19]:


agent=RLAgent(action_space=env.action_space,observation_space=env.observation_space,
              training_steps=training_steps,verbose=1)


# In[20]:


agent.debug=False
agent.use_memory=True


# In[21]:


agent.rewL=[]
agent.tot_rew=0


# In[22]:


agent.start()


# In[23]:


world=GenWorld(env=env)


# In[24]:


worldthread=Thread(name='world',target=world.run,args=(agent,2000,200))


# In[25]:


worldthread.start()


# In[26]:


# world.run(agent,n_episodes=2000,episode_maxlen=200)


# In[27]:


from matplotlib import pyplot as plt


# In[28]:


testing_len=len([agent.rewL[t] for t in world.test_episodes])


# In[30]:


# testing_len


# In[36]:


# agent.rewL


# In[33]:


# print(np.gradient(agent.rewL).mean())
print(agent.rewL)


# In[34]:


# plt.plot(np.gradient(agent.rewL))


# In[ ]:


episodes = 500
rewL=[]
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    steps=0
    while not done and steps<=200:
        # env.render()
        action,_ = agent.model.predict(state)
        state, reward, done, info = env.step(action)
        score+=reward
        steps+=1
    # print('Episode:{} Score:{}'.format(episode, score))
    rewL+=[score]
env.close()


# In[ ]:


from matplotlib import pyplot as plt
import numpy as np


# In[ ]:


np.array(rewL).mean()


# In[ ]:


plt.plot(rewL)


# In[ ]:


# get_ipython().run_line_magic('pinfo2', 'PPO')


# In[ ]:


len(agent.logL)


# In[ ]:




